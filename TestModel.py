import os, sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from report import build_test_summaries
from Unpack import unpack



def test(sess, env, args, actor, critic, desired_goal_dim, achieved_goal_dim, observation_dim):
    # Set path to save results
    tensorboard_dir = './' + args['env'] + '_' + args['name'] + '/test_' + datetime.now().strftime('%Y-%m-%d-%H')
    model_dir = './' + args['env'] + '_' + args['name'] + '/model'

    # add summary to tensorboard
    summary_ops, summary_vars = build_test_summaries()

    # initialize variables, create writer and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

    # restore session
    try:
        saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['name'] + '.ckpt'))
        print('-----------------------Start Test------------------------')
    except:
        print('WARNING: No model detected, Training first')
        sys.exit()

    # test in loop
    for i in range(int(args['episodes'])):

        # reset gym, get achieved_goal, desired_goal, state
        achieved_goal, desired_goal, s, s_prime = unpack(env.reset())
        episode_reward = 0

        for j in range(int(args['episode_length'])):

            # render episode
            if args['render']:
                env.render()

            # predict action
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))

            # play
            obs_next, reward, done, info = env.step(a[0])
            achieved_goal, desired_goal, state_next, state_prime_next = unpack(obs_next)

            # record reward and move to next state
            episode_reward += reward
            s = state_next

            # if episode ends
            if done:
                # write summary to tensorboard
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: episode_reward
                })
                writer.add_summary(summary_str, i)
                writer.flush()

                # print out results
                print('| Episode: {:d} | Reward: {:d}'.format(i, int(episode_reward)))

                break
    return