from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

DIR_PROCESSED = Path("data/processed")

# load metadata json file
with open(DIR_PROCESSED / "metadata.json", "r") as f:
    metadata = json.load(f) # load metadata file
print (f"Metadata is here: {metadata}")

# load csv file
train_df = pd.read_csv(DIR_PROCESSED / "train.csv")
print(f"Shape of training dataframe is: {train_df.shape}")
print(f"Head of training dataframe is: {train_df.head()}")

# show an example
example_row = np.load(train_df.iloc[0].path) # load path from first row - which loads the npz file
mel = example_row["mel_spec"] # get mel spectrogram in array form
label = example_row["label"]
print(f"Mel spectrogram shape: {mel.shape}, label: {label}")

# plot mel spectrogram
plt.imshow(mel, aspect="auto", origin="lower") # read spectrograms from low freq to high freq
plt.title(f"Mel Spectrogram - class: {label}")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar
plt.show()


