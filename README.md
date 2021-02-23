# Deep learning-based lung cancer detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/felixpeters/lung-cancer-detection/app/app.py)

[![Visualize in W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/felixpeters/lung-cancer-detection)

## Inspiration

> With an estimated 160,000 deaths in 2018, lung cancer is the most common cause of cancer death in the United States (Ardila et al. 2019).

> Lung cancer is one of the most prevalent cancers worldwide, causing 1.76 million deaths per year (Yu et al. 2020).

Clinical decision support systems have been developed to enable early diagnosis of lung cancer from CT images.
However, most of these tools are limited to lung or nodule segmentation, leaving classifation of nodules to the radiologist.
Early research shows that deep learning models can support with this task as well.
Integrating these research efforts into clinical applications is an active area of development.
See the [Arterys Marketplace](https://marketplace.arterys.com/) for examples of lung cancer detection models, some of which are currently under review for FDA or CE approval.
This project constitutes a design study of how a deep learning-based lung cancer detection app could look like.

## Dataset

The LIDC-IDRI dataset is used for training all models.
It can be accessed via the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).

## Roadmap

Nodule detection model:

- [ ] Preprocess LIDC dataset
- [ ] Train baseline model

Random ideas:

- Apply TCAV algorithm to trained model, use additional annotations as concepts

## References

### Materials

Basics of CT images:

- [Windowing](https://radiopaedia.org/articles/windowing-ct)
- [Hounsfield unit](https://radiopaedia.org/articles/hounsfield-unit)

PyTorch Lightning:

- [Docs](https://pytorch-lightning.readthedocs.io/en/stable/)

Monai (PyTorch-based library for medical imaging):

- [Docs](https://docs.monai.io/en/latest/)
- [MONAI Bootcamp YT playlist](https://www.youtube.com/playlist?list=PLtoSVSQ2XzyBro_Xs12cyerrGz4pEPylv)
- [Feature highlights](https://docs.monai.io/en/latest/highlights.html)
- [3D segmentation examples](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb)
- [DICOM loading example](https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb)

Preprocessing of LIDC-IDRI dataset:

Preprocessing of DICOM images:

- [DICOM Standard Browser](https://dicom.innolitics.com/ciods)
  - Useful for looking up meta data specification
- [Preprocessing code for LIDC dataset](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing)
  - Well documented
  - Creates masks from raw DICOM files
  - [Based on this repo](https://github.com/mikejhuang/LungNoduleDetectionClassification)
- Notebook series by Jeremy Howard
  - [Creating a metadata DataFrame](https://www.kaggle.com/jhoward/creating-a-metadata-dataframe-fastai)
  - [Some DICOM gotchas to be aware of](https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai/notebook)
  - [Don't see like a radiologist!](https://www.kaggle.com/jhoward/don-t-see-like-a-radiologist-fastai/notebook)
  - [Cleaning the data for rapid prototyping](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai/notebook)
- [Preprocessing of DSB 2017 dataset](https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial)

Lung cancer detection datasets:

- [LUNA 2016](https://luna16.grand-challenge.org/Home/)
- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [Data Science Bowl 2017](https://www.kaggle.com/c/data-science-bowl-2017/overview/description) (data can be found at academictorrents.com)
- [NLST](https://cdas.cancer.gov/learn/nlst/images/) (project proposal required)

### Scientific papers

Ardila, D., Kiraly, A. P., Bharadwaj, S., Choi, B., Reicher, J. J., Peng, L., Tse, D., Etemadi, M., Ye, W., Corrado, G., Naidich, D. P., and Shetty, S. 2019. “End-to-End Lung Cancer Screening with Three-Dimensional Deep Learning on Low-Dose Chest Computed Tomography,” Nature Medicine (25:6), Springer US, pp. 954–961. (https://doi.org/10.1038/s41591-019-0447-x).

Setio, A. A. A., Traverso, A., de Bel, T., Berens, M. S. N., Bogaard, C. van den, Cerello, P., Chen, H., Dou, Q., Fantacci, M. E., Geurts, B., Gugten, R. van der, Heng, P. A., Jansen, B., de Kaste, M. M. J., Kotov, V., Lin, J. Y. H., Manders, J. T. M. C., Sóñora-Mengana, A., García-Naranjo, J. C., Papavasileiou, E., Prokop, M., Saletta, M., Schaefer-Prokop, C. M., Scholten, E. T., Scholten, L., Snoeren, M. M., Torres, E. L., Vandemeulebroucke, J., Walasek, N., Zuidhof, G. C. A., Ginneken, B. van, and Jacobs, C. 2017. “Validation, Comparison, and Combination of Algorithms for Automatic Detection of Pulmonary Nodules in Computed Tomography Images: The LUNA16 Challenge,” Medical Image Analysis. (https://doi.org/10.1016/j.media.2017.06.015).

Svoboda, E. 2020. “Artificial Intelligence Is Improving the Detection of Lung Cancer,” Nature (587:7834), pp. S20–S22. (https://doi.org/10.1038/d41586-020-03157-9).

Yu, K. H., Lee, T. L. M., Yen, M. H., Kou, S. C., Rosen, B., Chiang, J. H., and Kohane, I. S. 2020. “Reproducible Machine Learning Methods for Lung Cancer Detection Using Computed Tomography Images: Algorithm Development and Validation,” Journal of Medical Internet Research (22:8), pp. 1–11. (https://doi.org/10.2196/16709).
