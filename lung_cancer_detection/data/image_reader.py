from pathlib import Path
from typing import Union, Sequence, Tuple

import pandas as pd
import numpy as np
from monai.data.image_reader import ImageReader


class LIDCReader(ImageReader):
    """Load CT scans from the preprocessed LIDC-IDRI dataset.

    Args:
        data_dir (Path): Path to directory which contains three sub-directories:
            - `images`: Data volumes with patient IDs as filenames, saved in `npy` format
            - `masks`: Binary segmentation masks with patient IDs as filenames , saved in `npy` format
            - `meta`: Two metadata files `scans.csv` and `nodules.csv`

    Raises:
        ValueError: If `data_dir` does not have the structure specified above.
    """

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        df_path = data_dir / "meta/scans.csv"
        img_path = data_dir / "images"
        mask_path = data_dir / "masks"
        if not (df_path.exists() and img_path.exists() and mask_path.exists()):
            raise ValueError("Data directory has invalid structure.")
        self.meta_df = pd.read_csv(df_path, index_col="PatientID")

    def verify_suffix(self, filename: str) -> bool:
        """
        Verify whether the specified file format is supported by LIDCReader.

        Args:
            filename (Union[Sequence[str], str]): file name or a list of file names to read (should always be one file for LIDC dataset)

        Returns:
            bool: if file format is supported
        """
        if isinstance(filename, list):
            raise ValueError(
                "LIDCReader only supports individual files to be loaded.")
        return filename.endswith(".npy")

    def read(self, data: str) -> Tuple[np.ndarray, pd.Series]:
        if isinstance(data, list) or not (data.endswith("npy")):
            raise ValueError(
                "LIDCReader only supports individual npy files to be loaded.")
        img = np.load(self.data_dir/data)
        pat_id = data.split("/")[1].split(".")[0]
        meta = self.meta_df.loc[pat_id]
        return (img, meta)

    def get_data(self, img: Tuple[np.ndarray, pd.Series]):
        return
