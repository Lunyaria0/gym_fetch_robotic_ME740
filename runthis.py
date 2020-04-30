import time
import gym
import argparse
import numpy as np
import tensorflow as tf
from agent import Actor, Critic
from Train import train
from TestModel import test
from noise import Noise


# Main
def main(args):
    # Set path to save result
    gym_dir = './' + args['env'] + '_' + args['name'] + '/gym'

    # Set random seed for reproducibility
    np.random.seed(int(args['seed']))
    tf.compat.v1.set_random_seed(int(args['seed']))

    with tf.compat.v1.Session() as sess:

        # Load environment
        env = gym.make(args['env'])
        env.seed(int(args['seed']))

        # get size of action and state (i.e. output and input for the agent)
        obs = env.reset()
        observation_dim = obs['observation'].shape[0]
        achieved_goal_dim = obs['achieved_goal'].shape[0]
        desired_goal_dim = obs['desired_goal'].shape[0]
        assert achieved_goal_dim == desired_goal_dim

        # state size = observation size + goal size
        state_dim = observation_dim + desired_goal_dim
        action_dim = env.action_space.shape[0]
        action_highbound = env.action_space.high

        # create actor
        actor = Actor(sess, state_dim, action_dim, action_highbound,
                      float(args['actor_lr']), float(args['tau']),
                      int(args['batch_size']), int(args['hidden_size']))

        # create critic
        critic = Critic(sess, state_dim, action_dim,
                        float(args['critic_lr']), float(args['tau']),
                        float(args['gamma']),
                        actor.n_actor_vars,
                        int(args['hidden_size']))

        # noise
        actor_noise = Noise(mu=np.zeros(action_dim))

        # train the network
        if not args['test']:
            train(sess, env, args, actor, critic, actor_noise, desired_goal_dim, achieved_goal_dim, observation_dim)
        else:
            test(sess, env, args, actor, critic, desired_goal_dim, achieved_goal_dim, observation_dim)

        # close gym
        env.close()

        # close session
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters setting
    parser.set_defaults(env='FetchReach-v1')
    parser.add_argument('--actor-lr', help='actor learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic learning rate', default=0.001)
    parser.add_argument('--batch-size', help='batch size', default=256)
    parser.add_argument('--gamma', help='discount factor reward', default=0.99)
    parser.add_argument('--tau', help='target update tau', default=0.001)
    parser.add_argument('--memory-size', help='size of the replay memory', default=100000)
    parser.add_argument('--hidden-size', help='number of nodes in hidden layer', default=256)
    parser.add_argument('--episodes', help='episodes to train', default=1000)
    parser.add_argument('--episode-length', help='max length of 1 episode', default=1000)
    parser.add_argument('--seed', help='random seed', default=1)
    parser.add_argument('--render', help='render the gym env', action='store_true')
    parser.add_argument('--test', help='test mode does not do exploration', action='store_true')
    parser.add_argument('--name', help='model name', default='L.L.')
    parser.set_defaults(render=False)
    parser.set_defaults(test=True)

    # parse arguments
    args = vars(parser.parse_args())

    # run main
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
