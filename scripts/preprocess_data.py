from pathlib import Path

from lung_cancer_detection.data.preprocessing import preprocess_lidc

data_dir = Path("/Volumes/LaCie/data/lung-cancer-detection/lidc-idri")
src_dir = data_dir/"LIDC-IDRI"
dest_dir = data_dir/"processed"

if __name__ == "__main__":
    preprocess_lidc(src_dir, dest_dir)
