import sys
from pathlib import Path
sys.path.append(str((Path.cwd()) / "src")) # have to add path for sys to recognize it

from data import dataloader_make
train, val, test = dataloader_make(bs=6, num_workers=0, pin=False)
input, target = next(iter(train)) # input is a stack of spectrograms, 
#target is a 1D tensor assigning each spectrogram a class

print(input.shape, target.shape) 

