# A two-stage method for spectral–spatial classification of hyperspectral images

This code is an implementation of the two-stage method, where the first stage is a \nu-SVM and the second stage is a L1 & L2-norm optimization that denoise the prediction results. The code can be used as a semisupervised per-pixel segmentation with smoothness, which is capable for multispectral/hyperspectral datasets. 
If you are using remote sensing datasets, for example, derived from Google Earth Engine, that may have 3 to 14 spectral bands available, the code gives an option to uplifting the data using the Lab color space information. This is proved to be useful in Reference 1 & 3.

The code of the first stage can be used as a classifier that fits the data, by training the \nu-SVM with a small set of available labels.

Notes:
- The code is written in MATLAB, and you need a 64-bit Windows computer to run the code.
- Contact: kangnicui2@gmail.com

## References
If you found the code useful, please cite:
- **Hyperspectral Change Detection Paper, In Preparation.** To run a demo, please run the RegionMaster.m, and then select Region as 1, data as RGB data, and task as binary classification.
- **Chan, R. H., Kan, K. K., Nikolova, M., & Plemmons, R. J.** (2020). A two-stage method for spectral–spatial classification of hyperspectral images. Journal of Mathematical Imaging and Vision, 62(6), 790-807. Link: https://link.springer.com/article/10.1007/s10851-019-00925-9.

If you applied the uplifting for multispectral images (e.g., from GEE), please cite:
- **Cai, X., Chan, R., Nikolova, M., & Zeng, T.** (2017). A three-stage approach for segmenting degraded color images: Smoothing, lifting and thresholding (SLaT). Journal of Scientific Computing, 72(3), 1313-1332. Link: https://link.springer.com/article/10.1007%2Fs10915-017-0402-2.
