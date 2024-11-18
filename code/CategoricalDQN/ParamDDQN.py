import os

EPISODES = 1000

BATCH_SIZE = 16
GAMMA = 0.09
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 400000

INITIAL_MEMORY = 16
MEMORY_SIZE = 2 * INITIAL_MEMORY

MODEL_SAVE_PATH = os.getcwd()+'/../rf-models/x2trainDistributional/'
MODEL_STATS_PATH = os.getcwd()+"/../stats/DistributionalDQNtr/"
