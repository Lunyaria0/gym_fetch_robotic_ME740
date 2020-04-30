from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#function to open log and get reward

def getReward(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    series = ea.scalars.Items('Reward')
    rewards = [s.value for s in series]
    tt = pd.DataFrame(rewards, columns=['Episode'])
    df = pd.DataFrame(rewards, columns=['Reward'])
    df['MeanReward'] = df['Reward'].rolling(window=100, min_periods=2).mean()
    df['SD'] = df['Reward'].rolling(window=200, min_periods=2).std()
    return df

# load
log_dir = 'FetchReach-v1_L.L.'
train_paths = [p for p in os.listdir(log_dir) if p.startswith('train_')]
train_paths.sort()
test_paths = [p for p in os.listdir(log_dir) if p.startswith('test_')]
test_paths.sort()

# look at first training session, first test session
df_train = getReward(os.path.join(log_dir, train_paths[0]))
df_test = getReward(os.path.join(log_dir, test_paths[0]))

# plot reward
sns.set_style('darkgrid')
sns.set_context("talk")
f, axs = plt.subplots(1, 2, figsize=(15, 5))
sns.tsplot(df_train['MeanReward'], ax=axs[0])
sns.tsplot(df_test['MeanReward'], ax=axs[1])
axs[0].set_title('256nodesV(1000epi) -- Training')
axs[1].set_title('256nodesV(1000epi) -- Testing')

plt.ylim(-50, 0)
plt.show()

