# A two-stage method for spectral–spatial classification of hyperspectral/multispectral images

This code is an implementation of the two-stage method, where the first stage is a \nu-SVM and the second stage is a L1 & L2-norm optimization that denoise the prediction results [2]. The code can be used as a semisupervised per-pixel segmentation with smoothness, which is capable for multispectral and hyperspectral datasets, with applications to change recognition and species classification, etc. 

If you are using remote sensing datasets, for example, derived from Google Earth Engine, that may have 3 to 14 spectral bands available, the code gives an option to uplifting the data using the Lab color space information. This is verified to be useful in [1,3].

The code of the first stage can be used as a classifier that fits the data alone, by training the \nu-SVM with a small set of available labels.

Notes:
- The code is written in MATLAB, and you need a 64-bit Windows computer to run the code ([LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)).
- To run a demo, please run the RegionMaster.m, and then select Region as 1, Data as RGB data, and Task as binary classification.
- To apply the code on your dataset, you should define a few inputs: HSI (dataset, m * n * p), Y2d (label, m * n), K_Known (No. of classes), trial_num (No. of trials).
- Contact: kangnicui2@gmail.com

## References
If you found the code useful, please cite:

- [1] **Change Recognition of Mining Ponds in Multispectral Imagery by Comparing Supervised and Semi-supervised Learning, In Preparation.** 

- [2] **Chan, R. H., Kan, K. K., Nikolova, M., & Plemmons, R. J.** (2020). A two-stage method for spectral–spatial classification of hyperspectral images. Journal of Mathematical Imaging and Vision, 62(6), 790-807. [Link](https://link.springer.com/article/10.1007/s10851-019-00925-9).

If you applied the uplifting for multispectral images (e.g., from GEE), please cite:

- [3] **Cai, X., Chan, R., Nikolova, M., & Zeng, T.** (2017). A three-stage approach for segmenting degraded color images: Smoothing, lifting and thresholding (SLaT). Journal of Scientific Computing, 72(3), 1313-1332. [Link](https://link.springer.com/article/10.1007%2Fs10915-017-0402-2).
