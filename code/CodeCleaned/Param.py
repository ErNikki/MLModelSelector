import os

#EPISODES = 1000
EPISODES = 250
EPISODES_LENGHT=1024
BATCH_SIZE = 4
GAMMA = 0.1
EPS_START = 1
EPS_END = 0.1
#EPS_DECAY = 100000
EPS_DECAY = 150000

INITIAL_MEMORY = 1024
MEMORY_SIZE = 2 * INITIAL_MEMORY
WARMUP=1024
UPDATE_TARGET=512
PLOT_STATS=10024
REDUCE_LR_STEPS=80

PATH_DATASET_FOLDER = os.getcwd() + "/../../dataset/SUN397/"
PATH_TRAIN_FOLDER=PATH_DATASET_FOLDER+  "train_selector"
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"
PATH_VAL_FOLDER=PATH_DATASET_FOLDER+"/val_selector"
PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"

#MODEL_SAVE_PATH = './rf-models/newTrainx2Regnet400newReward/'
#MODEL_SAVE_PATH = './rf-models/regnet400/'


"""
V1
e1df95eb7e35
if(label==predicted_label):
            if action==0:
                reward=1
                
            elif action==1:
                reward=0.7
                
            elif action==2:
                reward=0.5     
            
        else:
            reward=-1
"""
#1
#MODEL_SAVE_PATH = './rf-models/efficientnet_b3/TrainEnvV1/'
#MODEL_STATS_PATH = './rf-models/efficientnet_b3/TrainEnvV1/stats/'
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV1/test_stats/val_set/"
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV1/test_stats/test_set/"

"""
V2=b615b0309e5f
V3=91dc069b7058

if(label==predicted_label):
            if action==0:
                reward=1
                
            elif action==1:
                reward=0.7
                
            elif action==2:
                reward=0.5
            
            
        else:
            if action==0:
                reward=-1
                
            elif action==1:
                reward=-0.7
                
            elif action==2:
                reward=-0.5
"""
#MODEL_SAVE_PATH = './rf-models/efficientnet_b3/TrainEnvV2/'
#MODEL_STATS_PATH = './rf-models/efficientnet_b3/TrainEnvV2/stats/'
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV2/test_stats/val_set/"
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV2/test_stats/test_set/"

#MODEL_SAVE_PATH = './rf-models/efficientnet_b3/TrainEnvV3/'
#MODEL_STATS_PATH = './rf-models/efficientnet_b3/TrainEnvV3/stats/'
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV3/test_stats/val_set/"
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV3/test_stats/test_set/"

#MODEL_SAVE_PATH = './rf-models/efficientnet_b3/TrainEnvV4/'
#MODEL_STATS_PATH = './rf-models/efficientnet_b3/TrainEnvV4/stats/'
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV4/test_stats/val_set/"
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/TrainEnvV4/test_stats/test_set/"

#2 2fc2c6b34128
#MODEL_SAVE_PATH = './rf-models/efficientnet_b3/newTrainEnvV2/'
#MODEL_STATS_PATH = './rf-models/efficientnet_b3/newTrainEnvV2/stats/'
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/newTrainEnvV2/test_stats/val_set/"
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/newTrainEnvV2/test_stats/test_set/"

#3 57a2511dcccd
#MODEL_SAVE_PATH = './rf-models/efficientnet_b3/newTrainEnvSimplifiedV2/'
#MODEL_STATS_PATH = './rf-models/efficientnet_b3/newTrainEnvSimplifiedV2/stats/'
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/newTrainEnvSimplifiedV2/test_stats/val_set/"
#MODEL_STATS_TEST_PATH = "./rf-models/efficientnet_b3/newTrainEnvSimplifiedV2/test_stats/test_set/"

#POLICY GRADIENT<ZX 
#V3 = 0.9 a0db0078efef
MODEL_SAVE_PATH ="./rf-models/efficientnet_b3/policy_gradientV3/"
MODEL_STATS_PATH = "./rf-models/efficientnet_b3/policy_gradientV3/stats/"
MODEL_STATS_TEST_PATH="./rf-models/efficientnet_b3/policy_gradientV3/stats/test/"

# V4=0.1 1c2d1c664eb2
#MODEL_SAVE_PATH ="./rf-models/efficientnet_b3/policy_gradientV4/"
#MODEL_STATS_PATH = "./rf-models/efficientnet_b3/policy_gradientV4/stats/"
#MODEL_STATS_TEST_PATH="./rf-models/efficientnet_b3/policy_gradientV4/stats/test/"
