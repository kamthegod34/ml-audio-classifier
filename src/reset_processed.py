from pathlib import Path
import shutil

DIR_PROCESSED = Path("data/processed")
NPZ_DIR = DIR_PROCESSED / "npz"

def reset_processed():
    """deletes processed directory all full reset"""
    if NPZ_DIR.exists():
        shutil.rmtree(NPZ_DIR)
        print(f"deleted {NPZ_DIR} and all its contents")
    else:
        print(f"{NPZ_DIR} does not exist so there is nothing to delete")

    for folderName in ["train.csv", "val.csv", "test.csv", "metadata.json"]:
        (DIR_PROCESSED / folderName).unlink(missing_ok=True) # changed in v3.8 according to docs just lyk
        print(f"Deleted {folderName} (if it was present)")

def main():
    reset_processed()
    print("done resetting processed directory and its contents, yaay!")

if __name__ == "__main__":
    main()

        
        