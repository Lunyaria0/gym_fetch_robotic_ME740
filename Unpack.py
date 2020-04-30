import numpy as np


def unpack(obs):
    return obs['achieved_goal'], obs['desired_goal'], np.concatenate((obs['observation'], obs['desired_goal'])), np.concatenate((obs['observation'], obs['achieved_goal']))
