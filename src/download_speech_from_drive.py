from pathlib import Path
import shutil
import zipfile
import sys
import gdown
from pydub import AudioSegment

# this folder is the one that contains the speech files on google drive
# dont change this unless you want a different dataset
FOLDER_ID = "1f1JmfahXAwQs0ibQMRnOxJ_z5qKBFNPD"

DEST = Path("data/raw/speech")
TEMP_DEST = Path("data/temp/speech_drive")

AUDIO_EXTENSIONS = {".wav", ".mp3"}

def extract_zip_files(folder : Path):
    zips = list(folder.rglob("*.zip"))
    if not zips:
        print("no zip files found")
        return
    for z in zips:
        try:
            with zipfile.ZipFile(z, "r") as zip_file:
                zip_file.extractall(folder)

            z.unlink() # delete zip path after use
        except Exception as e:
            print(f"failed to extract zip file due to: {e}")
            continue

def collect_audio_files(folder : Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    return files

def main():

    DEST.mkdir(exist_ok=True) # dont crash if file exists
    TEMP_DEST.mkdir(exist_ok=True)

    gdown.download_folder(id=FOLDER_ID, output=str(TEMP_DEST), quiet=False, use_cookies=False)

    # call function to get audio files
    extract_zip_files(TEMP_DEST)

    all_files = collect_audio_files(TEMP_DEST)
    if not all_files:
        print("no audio files found, nooooo :(")
        return
    
    index = 1 
    for files in all_files:
        try:
            if files.suffix.lower() == ".mp3":
                audio = AudioSegment.from_mp3(files)
                audio.export(DEST / f"speech_{index:04d}.wav", format="wav")
            elif files.suffix.lower() == ".wav":
                shutil.copy(files, DEST / f"speech_{index:04d}.wav")
            index += 1
        except Exception as e:
            print(f"failed to convert file to wav due to: {e}")
            continue
    
    print("converted files to wav and placed them in data/raw/speech!!!!!!")

    
if __name__ == "__main__":
    main()
            

