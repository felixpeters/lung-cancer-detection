from pathlib import Path
from typing import Union, Sequence, Tuple, Dict, Any

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
        """Read image data from specified file and extract meta data from DataFrame.

        Args:
            data (str): Path to image file. Needs to include folder, i.e., `images` or `masks`

        Raises:
            ValueError: If list of filenames or filename with invalid suffix (i.e., not `npy`) is provided.

        Returns:
            Tuple[np.ndarray, pd.Series]: Numpy array with image data and meta data as series object
        """
        if isinstance(data, list) or not (data.endswith("npy")):
            raise ValueError(
                "LIDCReader only supports individual npy files to be loaded.")
        img = np.load(self.data_dir/data)
        pat_id = data.split("/")[1].split(".")[0]
        meta = self.meta_df.loc[pat_id]
        return (img, meta)

    def get_data(self, img: Tuple[np.ndarray, pd.Series]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract data array and meta data from loaded image and return them.

        Args:
            img (Tuple[np.ndarray, pd.Series]): Loaded image (typically output from `read` method)

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Numpy array of image data and dict of meta data.
        """
        img_array, scan_meta = img
        meta = self._get_meta_dict(scan_meta)
        meta["original_affine"] = self._get_affine(scan_meta)
        meta["affine"] = meta["original_affine"].copy()
        meta["spatial_shape"] = self._get_spacial_shape(img_array)
        return img_array, meta

    def _get_meta_dict(self, img_meta: pd.Series) -> Dict[str, Any]:
        """Extract ITK-compatible meta data from image meta.

        Args:
            img_meta (pd.Series): Image meta data (typically returned from `read` method)

        Returns:
            Dict[str, Any]: Meta data dictionary with keys origina, spacing and direction.
        """
        meta = {}
        meta["origin"] = np.asarray(
            [float(i.strip()) for i in img_meta.ImagePositionPatient[1:-1].split(",")])
        meta["spacing"] = np.asarray(
            [img_meta.PixelSpacing, img_meta.PixelSpacing, img_meta.SliceSpacing])
        meta["direction"] = np.eye(3)
        return meta
