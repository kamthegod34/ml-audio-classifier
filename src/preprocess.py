from __future__ import annotations
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import argparse

DIR_RAW = Path("data/raw")
DIR_PROCESSED = Path("data/processed")
NPZ_DIR = DIR_PROCESSED / "npz"
CLASSES = ["car", "drone", "speech"]

# audio parameters
SAMPLE_RATE = 16000
DURATION = 2.0
N_FFT = 1024 # frequency detail
N_MELS = 64 # frequency bins matching human hearing
HOP_LENGTH = 512 # how often you move the sample
RANDOM_STATE = 69

def audio_load(path: Path, sr: int=SAMPLE_RATE) -> np.ndarray:
    """ load single channel audio file at correct sample rate """
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def verify_raw_directory():
    missing = [cls for cls in CLASSES if not (DIR_RAW / cls).exists()]
    if missing:
        raise SystemExit(f"missing class directories in {DIR_RAW}, please add necessary folders for {missing}")
    counts = {cls: len(list((DIR_RAW / cls).glob("*.wav"))) for cls in CLASSES} # makes dictionary of .wav files in each class
    if any(v == 0 for v in counts.values()):
        raise SystemExit(f"no .wav files found in atleast one class directory in {DIR_RAW},"
                          f"there are scripts to process files if need them")
    return counts
        

def choppinator(y: np.ndarray, sr: int=SAMPLE_RATE, duration: float=DURATION) -> list[np.ndarray]:
    """ chop audio into duration length chunks """
    total_samples = int(sr * duration)
    if len(y) <= total_samples:
        return [librosa.util.fix_length(y, size=total_samples)]
    
    chunks = []
    n = len(y) // total_samples # number of full chunks
    for i in range(n):
        chunks.append(y[i*total_samples:(i+1)*total_samples]) # slice for all available full chunks
    remainder = len(y) - n * total_samples
    if remainder >= int(sr * 0.5):
        endPortion = y[-remainder:] # get the last portion of what is left - start from the end
        endPortionPadded= librosa.util.fix_length(endPortion, size=total_samples) # add remainder if at least 0.5 seconds
        chunks.append(endPortionPadded)
    return chunks

def make_mel_spectrogram(y: np.ndarray, sr: int=SAMPLE_RATE) -> np.ndarray:
    """ create mel spectrogram from audio chunk normalized to 0-1 """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)
    S_dB = librosa.power_to_db(S, ref=np.max) # dB relative to peak power

    # to ensure everything is between 0 and 1
    # we find the max and min and scale it to that range
    min_val = S_dB.min()
    max_val = S_dB.max()
    S_dB_norm = (S_dB - min_val) / (max_val - min_val + 1e-8 ) # add tiny number to avoid div by zero if max=min
    return S_dB_norm.astype(np.float32)

# create the row boilerplate that will be used to store one row of metadata per chunk
@dataclass 
class ExampleRow:
    id: str
    label: str
    path: str
    src: str

# creates structured dataset of all chunks and saves them to npz files
def generate_example_rows() -> list[ExampleRow]:
    NPZ_DIR.mkdir(parents=True, exist_ok=True) # ensure output directory exist if not, create it
    rows = []
    for cls in CLASSES:
        for wav in sorted((DIR_RAW / cls).glob("*.wav")):
            y = audio_load(wav, sr=SAMPLE_RATE)
            for chop in choppinator(y, sr=SAMPLE_RATE, duration=DURATION):
                mel_spec = make_mel_spectrogram(chop, sr=SAMPLE_RATE)
                id_chunk = uuid.uuid4().hex # .hex cause its cooler :)
                output_path = NPZ_DIR / f"{id_chunk}.npz"
                np.savez_compressed(output_path, mel_spec=mel_spec, label=cls)
                rows.append(ExampleRow(
                    id=id_chunk,
                    label=cls,
                    path=str(output_path),
                    src=str(wav)
                ))
    return rows

def enough_chunks_check(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    if (counts < 2).any(): # use .any() cause its a series, pandas thing
        raise SystemExit(f"not enough chunks in atleast one class to properly stratify: reduce duration or add more data")
    

def split_code(rows: list[ExampleRow]) -> dict:  # return metadata dictionary
    if not rows:
        raise SystemExit("no audio data found, please put wav files in data/raw/<cls>")
    df = pd.DataFrame([r.__dict__ for r in rows]) # cant be object so convert to dict first

    enough_chunks_check(df) # check if enough chunks are in each clas to stratify

    # split into train, validation and test sets
    train_df,temp_df  = train_test_split(df, test_size=0.3, train_size=0.7, random_state=RANDOM_STATE, 
    stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, train_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label"])

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DIR_PROCESSED / "train.csv", index=False) # index is true by default, have to change it, keep csv clean
    val_df.to_csv(DIR_PROCESSED / "val.csv", index=False)
    test_df.to_csv(DIR_PROCESSED / "test.csv", index=False)

    # important to debug and gives extra info
    counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "class_train": train_df["label"].value_counts().to_dict(), # dictionary in dictionary lol, cause series has exta info 
        "class_val": val_df["label"].value_counts().to_dict(),
        "class_test": test_df["label"].value_counts().to_dict(),
    }

    # make a metadata file so if i forget what i did i can check on it
    # json cause its very leggible
    metadata = {
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "random_state": RANDOM_STATE,
        "classes": CLASSES,
        "counts": counts
    }

    with open(DIR_PROCESSED / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess audio data into mel spectrograms and then " \
    "split that into train/val/test sets")
    parser.add_argument("--raw", default="data/raw", type=str, help="path to raw audio files")
    parser.add_argument("--processed", default="data/processed", type=str, help="path to save processed files")
    parser.add_argument("--sr", default=SAMPLE_RATE, type=int, help="sample rate to load audio")
    parser.add_argument("--duration", default=DURATION, type=float, help="duration of each audio chunk in seconds")
    parser.add_argument("--n_fft", default=N_FFT, type=int, help="number of fft components")
    parser.add_argument("--n_mels", default=N_MELS, type=int, help="number of mel bands to generate")
    parser.add_argument("--hop_length", default=HOP_LENGTH, type=int, help="hop length, number of samples to move forward after each window")
    parser.add_argument("--random_state", default=RANDOM_STATE, type=int, help="helps with reproducibility of splits - default is 69 :)")
    parser.add_argument("--classes", default=CLASSES, nargs="+", type=str, help="list of class names - should match folders in subdirectory")
    return parser.parse_args()

def main():

    # override the globals with user input
    global DIR_RAW, DIR_PROCESSED, NPZ_DIR, SAMPLE_RATE, DURATION, N_FFT, N_MELS, HOP_LENGTH, RANDOM_STATE, CLASSES

    # get the values that the user inputted from the CLI or it will get the defaults
    DIR_RAW = Path(args.raw)
    DIR_PROCESSED = Path(args.processed)
    NPZ_DIR = DIR_PROCESSED / "npz" # have to recompute npz files after DIR_PROCESSED changes

    # override the globals
    SAMPLE_RATE = args.sr
    DURATION = args.duration
    N_FFT = args.n_fft    
    N_MELS = args.n_mels
    HOP_LENGTH = args.hop_length
    RANDOM_STATE = args.random_state
    CLASSES = list(args.classes)

    print(f"--preprocess-- sample rate: {SAMPLE_RATE}, duration: {DURATION}, n_fft: {N_FFT},"
          f" n_mels: {N_MELS}, hop_length: {HOP_LENGTH}")
    print(f"--preprocess-- raw dir: {DIR_RAW} and processed dir: {DIR_PROCESSED}")

    verify_raw_directory() # make sure raw directory is ready to go
    # for debugging - variable (counts) is returned, might be helpful just lyk

    rows = generate_example_rows()
    metadata = split_code(rows)
    
    print(json.dumps(metadata["counts"], indent=2 ))
    print(f"--preprocess-- done! -> {DIR_PROCESSED}")



if __name__ == "__main__":
    args = parse_args()
    main()