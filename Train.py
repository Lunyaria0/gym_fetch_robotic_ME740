import os
from Memory import Memory
import numpy as np
import tensorflow as tf
from datetime import datetime
from report import build_summaries
from Unpack import unpack


def train(sess, env, args, actor, critic, actor_noise, desired_goal_dim, achieved_goal_dim, observation_dim):
    # Set path to save results
    tensorboard_dir = './' + args['env'] + '_' + args['name'] + '/train_' + datetime.now().strftime('%Y-%m-%d-%H')
    model_dir = './' + args['env'] + '_' + args['name'] + '/model'

    # add summary to tensorboard
    summary_ops, summary_vars = build_summaries()

    # initialize variables, create writer and saver
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    writer = tf.compat.v1.summary.FileWriter(tensorboard_dir, sess.graph)

    # restore session if exists
    try:
        saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['name'] + '.ckpt'))
        print('------------------------Continue--------------------------')
    except:
        print('----------------------New Training------------------------')

    # initialize target network weights and replay memory
    actor.update()
    critic.update()
    replay_memory = Memory(int(args['memory_size']), int(args['seed']))

    # train in loop
    for i in range(int(args['episodes'])):

        # reset gym, get achieved_goal, desired_goal, state
        achieved_goal, desired_goal, s, s_prime = unpack(env.reset())
        episode_reward = 0
        episode_maximum_q = 0

        for j in range(int(args['episode_length'])):

            # render episode
            if args['render']:
                env.render()

            # predict action and add noise
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))
            a = a + actor_noise.get_noise()

            # play
            obs_next, reward, done, info = env.step(a[0])
            achieved_goal, desired_goal, state_next, state_prime_next = unpack(obs_next)

            # add normal experience to memory
            replay_memory.add(np.reshape(s, (actor.state_dim,)),
                              np.reshape(a, (actor.action_dim,)),
                              reward,
                              done,
                              np.reshape(state_next, (actor.state_dim,)))

            # add hindsight experience to memory
            substitute_goal = achieved_goal.copy()
            substitute_reward = env.compute_reward(achieved_goal, substitute_goal, info)
            replay_memory.add(np.reshape(s_prime, (actor.state_dim,)),
                              np.reshape(a, (actor.action_dim,)),
                              substitute_reward,
                              True,
                              np.reshape(state_prime_next, (actor.state_dim,)))

            # start to train when there's enough experience
            if replay_memory.size() > int(args['batch_size']):
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_memory.sample_batch(int(args['batch_size']))

                # find TD -- temporal difference
                # actor find target action
                a2_batch = actor.predict_target(s2_batch)

                # critic find target q
                q2_batch = critic.predict_target(s2_batch, a2_batch)

                # add a decay of q to reward if not done
                r_batch_discounted = []
                for k in range(int(args['batch_size'])):
                    if d_batch[k]:
                        r_batch_discounted.append(r_batch[k])
                    else:
                        r_batch_discounted.append(r_batch[k] + critic.gamma * q2_batch[k])

                # train critic with state, action, and reward
                pred_q, _ = critic.train(s_batch,
                                         a_batch,
                                         np.reshape(r_batch_discounted, (int(args['batch_size']), 1)))

                # record maximum q
                episode_maximum_q += np.amax(pred_q)

                # actor find action
                a_outs = actor.predict(s_batch)

                # get comment from critic
                comment_gradients = critic.get_comment_gradients(s_batch, a_outs)

                # train actor with state and the comment gradients
                actor.train(s_batch, comment_gradients[0])

                # Update target networks
                actor.update()
                critic.update()

            # record reward and move to next state
            episode_reward += reward
            s = state_next

            # if episode ends
            if done:
                # write summary to tensorboard
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: episode_reward,
                    summary_vars[1]: episode_maximum_q / float(j)
                })
                writer.add_summary(summary_str, i)
                writer.flush()

                # print out results
                print('| Episode: {:d} | Reward: {:d} '.format(i, int(episode_reward)))
                # save model
                saver.save(sess, os.path.join(model_dir, args['env'] + '_' + args['name'] + '.ckpt'))

                break
    return