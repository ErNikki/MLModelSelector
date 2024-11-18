import os
import numpy as np
import pandas as pd 

def data_to_df(data_dir, subset=None, validation_split=None):
    df = pd.DataFrame()
    filenames = []
    labels = []
    
    
    
    for dataset in os.listdir(data_dir):
        img_list = os.listdir(os.path.join(data_dir, dataset))
        label = name_to_idx[dataset]
        
        for image in img_list:
            filenames.append(os.path.join(data_dir, dataset, image))
            labels.append(label)
        
    df["filenames"] = filenames
    df["labels"] = labels
    
    """
    if subset == "train":
        split_indexes = int(len(df) * validation_split)
        train_df = df[split_indexes:]
        val_df = df[:split_indexes]
        return train_df, val_df
    """
    
    return df

train_df, val_df = data_to_df(train_dir, subset="train", validation_split=0.2)