import os
from pathlib import Path
from functools import reduce

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import cm

DATA_DIR = Path(__file__).absolute().parent / "data"

@st.cache
def load_meta():
    scan_df = pd.read_csv(DATA_DIR / "scan_meta.csv")
    nod_df = pd.read_csv(DATA_DIR / "nodule_meta.csv")
    return scan_df, nod_df

@st.cache
def load_raw_img(pid):
    img = np.load(DATA_DIR/pid/"scan.npy")
    return img

@st.cache
def load_mask(pid):
    fnames = sorted((DATA_DIR/pid).glob('*_mask.npy'))
    masks = [np.load(fname) for fname in fnames]
    mask = reduce(np.logical_or, masks)
    return mask

@st.cache(allow_output_mutation=True)
def get_img_slice(img, z, window=(-600, 1500)):
    # clip pixel values to desired window
    level, width = window
    img = np.clip(img, level-(width/2), level+(width/2))
    # normalize pixel values to 0-1 range
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)
    # convert to Pillow image for display
    img_slice = img[:,:,z]
    pil_img = Image.fromarray(np.uint8(cm.gray(img_slice)*255))
    return pil_img.convert('RGBA')

@st.cache
def get_overlay():
    arr = np.zeros((512, 512, 4)).astype(np.uint8)
    arr[:,:,1] = 128
    arr[:,:,3] = 128
    overlay = Image.fromarray(arr, mode='RGBA')
    return overlay

@st.cache
def get_mask_slice(mask, z):
    mask_slice = (mask[:,:,z]*96).astype(np.uint8)
    mask_img = Image.fromarray(mask_slice, mode='L')
    return mask_img

scan_df, nod_df = load_meta()
scan = scan_df.iloc[0]
pid = scan.PatientID

img_arr = load_raw_img(pid)
mask_arr = load_mask(pid)

st.header("Selected case for lung cancer detection application")

st.subheader("Patient information")

st.write("**Patient ID:**", scan.PatientID)
st.write("**Diagnosis:**", "Malignant, primary lung cancer")
st.write("**Diagnosis method:**", "Biopsy")

st.subheader(f"CT scan")

img_placeholder = st.empty()

col1, col2 = st.beta_columns(2)

with col1:
    st.write("**Pixel spacing**")
    st.write(f"x: {scan.PixelSpacing:.2f} mm")
    st.write(f"y: {scan.PixelSpacing:.2f} mm")
    st.write(f"z: {scan.SliceSpacing:.2f} mm")
    st.write("**Device**")
    st.write(f"{scan.ManufacturerModelName} (by {scan.Manufacturer})")

with col2:
    overlay_nodules = st.checkbox("Show nodule overlay", value=True)
    z = st.slider("Slice:", min_value=1, max_value=img_arr.shape[2], value=int(img_arr.shape[2]/2))
    level = st.number_input("Window level:", value=-600)
    width = st.number_input("Window width:", value=1500)

img = get_img_slice(img_arr, z-1, window=(level, width))

if overlay_nodules:
    mask = get_mask_slice(mask_arr, z-1)
    overlay = get_overlay()
    ct = Image.composite(overlay, img, mask)
    img_placeholder.image(ct, use_column_width=True)
else:
    img_placeholder.image(img, use_column_width=True)

st.subheader("Detected nodules")

st.write("See [this page](https://pylidc.github.io/annotation.html) for information about the coding scheme.")

nodules = nod_df.iloc[:, 2:]
st.write(nodules)
