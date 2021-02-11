import os
from pathlib import Path

import streamlit as st
import pandas as pd

DATA_DIR = Path(__file__).absolute().parent / "data"

@st.cache
def load_meta():
    scan_df = pd.read_csv(DATA_DIR / "scan_meta.csv")
    nod_df = pd.read_csv(DATA_DIR / "nodule_meta.csv")
    return scan_df, nod_df

def load_ct_img():
    return

scan_df, nod_df = load_meta()

st.header("Lung cancer detection")

st.write(scan_df.shape)

st.write(nod_df)
