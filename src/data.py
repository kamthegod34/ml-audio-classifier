from pathlib import Path
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from utils  import make_dataloader_seeding



PROJECT_ROOT = Path(__file__).resolve().parents[1] # project root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_2_IDX = {"drone": 0, "car":1, "speech": 2} # works as a translator, changing to numbers is better for math

class sample_example_loader(Dataset): # teaches how to load a sample
    def __init__(self, csv_path: str | Path, augment: bool=False):
        directory_possible = PROCESSED_DIR / csv_path
        csv_path = directory_possible if directory_possible.exists() else PROJECT_ROOT / csv_path

        self.df = pd.read_csv(csv_path)
        self.augment = augment
    
    def __len__(self): # dataloader relies on __len__, you have to make this distinction
        return len(self.df)
    
    def __getitem__(self, i: int):
        row = self.df.iloc[i] # gets a row
        npz_path = (PROJECT_ROOT / row["path"]).resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"File listed in CSV not found: {npz_path}")
        example_row = np.load(npz_path)
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
                start_high = max(1,time - 8 + 1)
                time_start = np.random.randint(0, start_high) # np slicing CAN go out of bounds, no problem
                x[..., time_start:time_start+8] = 0.0 # timemasking like SpecAugment
                # gaussian noise - random
                x = x + 0.01*torch.randn_like(x)
        return x, torch.tensor(y, dtype=torch.long)
    
def dataloader_make(  # adding some default arguments
        test_csv = "data/processed/test.csv",
        train_csv = "data/processed/train.csv",
        val_csv = "data/processed/val.csv",
        bs = 32, #  - noisier gradients  + smoother gradients
        num_workers = 2, # 2 prepare batches while GPU trains (no bottleneck)
        pin = True, # prevent paging - no movement into hard disk
        seed: int = 69,
        drop_last: bool = True # to prevent internal covariate shift ill default to true, 
                                # but for better reproducibility false is preferred
):
    
    # create generator and get function for worker init
    g, worker_init = make_dataloader_seeding(seed)

    # create instances of custom class
    train_ds = sample_example_loader(train_csv, augment=True)
    test_ds = sample_example_loader(test_csv, augment=False)
    val_ds = sample_example_loader(val_csv, augment=False)

    # feeders for training with correct settings and instructions - iterable dataset 
    train_dloader = DataLoader(train_ds, batch_size = bs, shuffle=True, drop_last=drop_last, num_workers=num_workers, 
                               pin_memory=pin, persistent_workers= num_workers > 0,
                               worker_init_fn=worker_init, generator=g)
    val_dloader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    test_dloader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)

    return train_dloader, val_dloader, test_dloader