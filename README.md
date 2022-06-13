# Change detection of multispectral images using SVM-STV with lifting.

This code is an implementation of the SVM-STV method, where the first stage is a \nu-SVM and the second stage is a L1 & L2-norm optimization that perform denoising [3]. The code can be used as a semisupervised per-pixel segmentation/classification algorithm with smoothing that is capable for multispectral datasets for change recognition and species classification, etc. 

If you are using remote sensing datasets (e.g., derived from Google Earth Engine) that may have RGB spectral bands available, the code gives an option to uplift the data using the Lab color space information. This is verified to be useful in [1,2,4], especially for RGB data.

The code of the first stage can be used as a classifier that fits the data alone, by training the \nu-SVM with a small set of available labels, see [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).

Notes:
- The code is mainly written in MATLAB, and you need a 64-bit Windows computer to run the code ([LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)).
- To run a demo, please run the SVMLSMaster.m, and then select Region as 1, Data as RGB data, and Task as binary classification. The data used in this study could be found here: [data](https://zenodo.org/record/6400211#.Yo23DqhByMo).
- To apply the code on your dataset, you should define a few inputs: HSI (dataset, m * n * p), Y2d (label, m * n), K_Known (No. of classes), trial_num (No. of trials).
- If you want to pre-train a model and then apply it to testing datasets, please revise the code in 'test' folder, where the TrainModel.m can be revised to pre-train the \nu-SVM in the first stage. Then choose proper denoising parameters (based on your experience during training): 'par', 'par2' in the second stage, and revise TestMaster to run the code on your testing datasets.
- Contact: kangnicui2@gmail.com

## References
If you found the code useful, please cite:

- [1] **Cui, K., Camalan, S., Li, R., Pauca, V. P., Alqahtani, S., Silman, S., Plemmons, R. J., Dethier, E. N., Lutz, D. A., Chan, R. H.**  (2022). Semi-supervised Change Detection of Small Water Bodies Using RGB and Multispectral Images in Peruvian Rainforests. *Proc IEEE Workshop Hyperspectral Image Signal Process Evol Remote Sens*, IEEE.

- [2] **Camalan, S., Cui, K., Pauca, V. P., Alqahtani, S., Silman, S., Chan, R. H., Plemmons, R. J., Dethier, E. N., Fernandez, L. E., Lutz, D. A.**  (2022). Change Detection of Amazonian Alluvial Gold Mining Using Deep Learning and Sentinel-2 Imagery. *Remote Sensing, 14(7), 1746*, MDPI. [Link](https://www.mdpi.com/2072-4292/14/7/1746/htm).

- [3] **Chan, R. H., Kan, K. K., Nikolova, M., & Plemmons, R. J.** (2020). A two-stage method for spectral–spatial classification of hyperspectral images. *Journal of Mathematical Imaging and Vision, 62(6), 790-807*, Springer. [Link](https://link.springer.com/article/10.1007/s10851-019-00925-9).

If you find the uplifting step useful, please cite:

- [4] **Cai, X., Chan, R., Nikolova, M., & Zeng, T.** (2017). A three-stage approach for segmenting degraded color images: Smoothing, lifting and thresholding (SLaT). *Journal of Scientific Computing, 72(3), 1313-1332*, Springer. [Link](https://link.springer.com/article/10.1007%2Fs10915-017-0402-2).
