import yaml
import argparse
from lung_cancer_detection.data.preprocessing import preprocess_lidc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


data_dir = Path("/Volumes/LaCie/data/lung-cancer-detection/lidc-idri")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script extracts volumes, masks and metadata from LIDC-IDRI lung cancer dataset.")
    parser.add_argument(
        '--config',
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parents[1] / "configs/baseline.yaml",
        help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    data_dir = Path(config["data"]["lidc_dir"]).absolute()
    src_dir = data_dir/"LIDC-IDRI"
    dest_dir = data_dir/"processed"
    print("CONFIGURATION:")
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dest_dir}")

    preprocess_lidc(src_dir, dest_dir)
