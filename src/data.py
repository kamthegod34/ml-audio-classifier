from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_2_IDX = {"drone": 0, "car":1, "speech": 2} # works as a translator, changing to numbers is better for math

class sample_example_loader(Dataset): # teaches how to load a sample
    def __init__(self, csv_path: str=Path, augment: bool=False):
        self.df = pd.read_csv("data/processed/tran.csv")
        self.augment = augment
    
    def __len__(self): # dataloader relies on __len__, you have to make this distinction
        return len(self.df)
    
    def __getitem__(self, i: int):
        row = self.df.iloc[i] # gets a row
        example_row = np.load(row.path) # loads the chunk (npz) file from where the row came from
        mel = example_row["mel_spec"].astype("float32") # no need for more precision and docs say its default
        # also returns spectrogram array in shape [n_mels, time_frames]


        # we could just use the .csv row from the i value, but for greater security get label from .npz file
        label_str= (example_row["label"].item()
                    # we are checking if label a "key" in example_row and if it can be converted into a python value
                    if "label" in example_row and hasattr(example_row["label"], "item")
                    else row.label # backup option
                    )
        y = CLASS_2_IDX[str(label_str)] # str() is acutally uneccesary since item() turns it to a string, just in case

        x = torch.from_numpy(mel).unsqueeze(0) # could also use [None,...]

        if self.augment:
            time = x.shape[-1]
            if time > 0:
                # time masking
                time_start = np.random.randint(0, max(1,time - 8)) # np slicing CAN go out of bounds, no problem
                x[..., time_start:time_start+8] *= 0.69 # relatively aggresive mask, for a project of this size
                # gaussian noise - random
                