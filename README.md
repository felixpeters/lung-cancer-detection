# Deep learning-based lung cancer detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/felixpeters/lung-cancer-detection/app/app.py)

## Dataset

The LIDC-IDRI dataset is used for training all models.
It can be accessed via the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).

## Roadmap

Model training:

- [ ] Preprocess LIDC dataset
- [ ] Train baseline model

## References

### Materials

PyTorch Lightning:

- [Docs](https://pytorch-lightning.readthedocs.io/en/stable/)

Monai (PyTorch-based library for medical imaging):

- [Docs](https://docs.monai.io/en/latest/)
- [MONAI Bootcamp YT playlist](https://www.youtube.com/playlist?list=PLtoSVSQ2XzyBro_Xs12cyerrGz4pEPylv)

Preprocessing of LIDC-IDRI dataset:

- [Older base repo](https://github.com/mikejhuang/LungNoduleDetectionClassification) 
- [More recent repo](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing)

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
