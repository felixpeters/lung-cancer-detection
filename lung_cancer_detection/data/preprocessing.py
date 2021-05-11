import math
import os
from functools import reduce
from pathlib import Path
from statistics import median_high
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pydicom as dicom
import pylidc as pl
from pylidc.utils import consensus
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def preprocess_lidc(src: Path, dest: Path, sample: Union[Sequence[str], bool] =
                    False, nod_size: Tuple[int] = (100, 100, 60)):
    """Preprocesses the LIDC-IDRI dataset after being downloaded from TCIA.

    Args:
        src (Path): Path to directory where the DICOM folders reside.
        dest (Path): Path to which volumes, masks and metadata should be written.
        sample_size (int): Sample size. Mainly used for testing. Defaults to False
        nod_size (Tuple[int]): Size of extracted nodule volumes. Defaults to (100, 100, 60) pixels.
    """
    img_path = dest / "images"
    img_path.mkdir(parents=True, exist_ok=True)
    mask_path = dest / "masks"
    mask_path.mkdir(parents=True, exist_ok=True)
    nod_path = dest / "nodules"
    nod_path.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)

    pids = get_pids(src)
    scan_data = []
    nod_data = []
    if sample:
        pids = sample

    for pid in tqdm(pids):
        scan = get_scan(pid)
        scan_meta = get_scan_meta(scan)
        scan_data.append(scan_meta)

        vol = scan.to_volume(verbose=False)
        np.save(img_path/f"{pid}.npy", vol.astype(np.int16))
        ann_clusters = scan.cluster_annotations(verbose=False)
        masks = [np.zeros(vol.shape, dtype=np.uint8)]

        for i, cluster in enumerate(ann_clusters):
            # pad whole image for segmentation mask
            pad_sz = int(np.max(vol.shape))
            _, bbox = consensus(cluster, ret_masks=False)
            mask, _ = consensus(cluster, ret_masks=False, pad=pad_sz)
            # calc padding for nodule volume
            nod_pad_sz = [(math.ceil(i/2), math.floor(i/2)) for i in
                          (np.array(nod_size) - np.array(vol[bbox].shape))]
            _, pbbox = consensus(cluster, ret_masks=False, pad=nod_pad_sz)
            nod_vol = vol[pbbox]
            np.save(nod_path/f"{pid}_{i}.npy", nod_vol.astype(np.int16))
            nod_meta = get_nod_meta(scan, cluster, i, bbox)
            nod_data.append(nod_meta)
            masks.append(mask)

        mask = reduce(np.logical_or, masks)
        np.save(mask_path/f"{pid}.npy", mask.astype(np.uint8))

    scan_df = pd.DataFrame(data=scan_data)
    scan_df.to_csv(meta_path/"scans.csv", index=False)
    nod_df = pd.DataFrame(data=nod_data)
    nod_df.to_csv(meta_path/"nodules.csv", index=False)
    return


def get_pids(dir: Path) -> Sequence[str]:
    """Extracts patient IDs from folder names in the given directory.

    Args:
        dir (Path): Source directory, typically 'LIDC-IDRI' folder when data is downloaded from TCIA.

    Returns:
        Sequence[str]: Array of patient IDs
    """
    pids = [f for f in os.listdir(dir) if not f.startswith(
        '.') and not f.endswith('.csv')]
    pids.sort()
    return pids


def get_scan(pid: str) -> pl.Scan:
    """Retrieves the first scan found for the given patient ID.

    Args:
        pid (str): Patient ID

    Returns:
        pl.Scan: Scan
    """
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    return scan


def get_scan_meta(scan: pl.Scan) -> Dict[str, Any]:
    """Extracts basic metadata from a Scan object, i.e., metadata that does not require opening the respective DICOM files.

    Args:
        scan (pl.Scan): The scan from where metadata should be extracted.

    Returns:
        Dict[str, Any]: A dictionary containing the metadata.
    """
    meta = {
        "StudyID": scan.study_instance_uid,
        "SeriesID": scan.series_instance_uid,
        "PatientID": scan.patient_id,
        "SliceThickness": scan.slice_thickness,
        "SliceSpacing": scan.slice_spacing,
        "PixelSpacing": scan.pixel_spacing,
        "ContrastUsed": scan.contrast_used,
    }
    dcm_meta = get_dcm_meta(scan)
    meta.update(dcm_meta)
    return meta


def get_nod_meta(scan: pl.Scan, cluster: Sequence[pl.Annotation], index: int, bbox: Tuple[slice]) -> Dict[str, Any]:
    """Extracts metadata from a given lung nodule.

    Args:
        scan (pl.Scan): Scan in which the nodule was detected.
        cluster (Sequence[pl.Annotation]): List of annotations for this nodule (typically derived from the pylidc.scan.cluster_annotations method)
        index (int): The index of the given nodule in this scan.
        bbox (Tuple[slice]): Bounding box of the given nodule (typically derived from pylidc.utils.consensus function).

    Returns:
        Dict[str, Any]: A dictionary containing the metadata.
    """
    meta = {
        'PatientID': scan.patient_id,
        'StudyID': scan.study_instance_uid,
        'SeriesID': scan.series_instance_uid,
        'NoduleID': index,
        'NumAnnotations': len(cluster),
        'Diameter': np.mean([ann.diameter for ann in cluster]),
        'SurfaceArea': np.mean([ann.surface_area for ann in cluster]),
        'Volume': np.mean([ann.volume for ann in cluster]),
        'Malignancy': median_high([ann.malignancy for ann in cluster]),
        'Texture': median_high([ann.texture for ann in cluster]),
        'Spiculation': median_high([ann.spiculation for ann in cluster]),
        'Lobulation': median_high([ann.lobulation for ann in cluster]),
        'Margin': median_high([ann.margin for ann in cluster]),
        'Sphericity': median_high([ann.sphericity for ann in cluster]),
        'Calcification': median_high([ann.calcification for ann in cluster]),
        'InternalStructure': median_high([ann.internalStructure for ann in cluster]),
        'Subtlety': median_high([ann.subtlety for ann in cluster]),
        'x_start': bbox[0].start,
        'x_stop': bbox[0].stop,
        'y_start': bbox[1].start,
        'y_stop': bbox[1].stop,
        'z_start': bbox[2].start,
        'z_stop': bbox[2].stop,
    }
    return meta


def get_dcm_meta(scan: pl.Scan) -> Dict[str, Any]:
    """Extracts metadata from a scan which requires loading the DICOM files.

    Args:
        scan (pl.Scan): The scan for which metadata should be extracted.

    Returns:
        Dict[str, Any]: A dictionary containing the metadata.
    """
    path = Path(scan.get_path_to_dicom_files())
    fnames = sorted([fname for fname in os.listdir(path)
                     if fname.endswith('.dcm')])
    dcm = dicom.dcmread(path/fnames[0])
    meta = {
        "ImagePositionPatient": getattr(dcm, "ImagePositionPatient", np.nan),
        "ImageOrientationPatient": getattr(dcm, "ImageOrientationPatient", np.nan),
        "Rows": getattr(dcm, "Rows", np.nan),
        "Columns": getattr(dcm, "Columns", np.nan),
        "RescaleIntercept": getattr(dcm, "RescaleIntercept", np.nan),
        "RescaleSlope": getattr(dcm, "RescaleSlope", np.nan),
        "WindowCenter": getattr(dcm, "WindowCenter", np.nan),
        "WindowWidth": getattr(dcm, "WindowWidth", np.nan),
        "BitsAllocated": getattr(dcm, "BitsAllocated", np.nan),
        "PixelRepresentation": getattr(dcm, "PixelRepresentation", np.nan),
        "Manufacturer": getattr(dcm, "Manufacturer", ""),
        "ManufacturerModelName": getattr(dcm, "ManufacturerModelName", ""),
    }
    return meta


def split_lidc(meta_path: Path, val_split: float = 0.2, seed: int = 47) -> Dict[str, Sequence[str]]:
    """Splits LIDC-IDRI dataset into training and validation sets by patient ID.

    Args:
        meta_path (Path): Path to CSV file containing scan metadata.
        val_split (float): Percentage of validation data. Defaults to 0.2.
        seed (int): Random seed to use for splitting. Defaults to 47.

    Returns:
        Dict[str, Sequence[str]]: Dictionary containg patient IDs for training and
        validation sets.
    """
    scans = pd.read_csv(meta_path/"scans.csv", index_col="PatientID")
    nods = pd.read_csv(meta_path/"nodules.csv", index_col="PatientID")
    pids = list(scans.index)
    train_pids, valid_pids = train_test_split(
        pids, test_size=val_split, random_state=seed, shuffle=True)
    train_dict = [{
        "pid": pid,
        "image": f"images/{pid}.npy",
        "mask": f"masks/{pid}.npy",
        "nodules": [{
            "nid": nid,
            "image": f"nodules/{pid}_{nid}.npy",
            "annotations": anns,
            "malignancy": mal,
        } for nid, anns, mal in
            zip(list(nods.query(f"PatientID == '{pid}'").NoduleID),
                list(nods.query(f"PatientID == '{pid}'").NumAnnotations),
                list(nods.query(f"PatientID == '{pid}'").Malignancy))],
    } for pid in train_pids]
    valid_dict = [{
        "pid": pid,
        "image": f"images/{pid}.npy",
        "mask": f"masks/{pid}.npy",
        "nodules": [{
            "nid": nid,
            "image": f"nodules/{pid}_{nid}.npy",
            "annotations": anns,
            "malignancy": mal,
        } for nid, anns, mal in
            zip(list(nods.query(f"PatientID == '{pid}'").NoduleID),
                list(nods.query(f"PatientID == '{pid}'").NumAnnotations),
                list(nods.query(f"PatientID == '{pid}'").Malignancy))],
    } for pid in valid_pids]
    return train_dict, valid_dict
