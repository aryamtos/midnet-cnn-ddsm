# MidNet Convolutional Network

## Arquitetura da Rede

A rede foi desenvolvida para detecção de anormalidades na mama, ou seja, massas benignas e malignas, após a detecção de nódulos mamários foi aplicada a classificação de imagens em benigna ou maligna.

Foram utilizadas operações convolutivas com kernel 3x3, como resultado da convolução, a imagem foi filtrada para características específicas. Após cada camada convolutiva foi aplicado o batchnormalization para padronização de variáveis de entradas brutas, sendo aplicada antes de cada função de ativação ReLU.  O foco da rede é utilizar ao máximo os recursos de regularização tendo em vista, reduzir o overfitting. Dentre as técnicas de regularização utilizadas estão:

- *Regularização L2*
- *Dropout*
- *Batchnormalization*

![Untitled](MidNet%20Convolutional%20Network/Untitled.png)

**Link Abaixo**

[midnet-cnn-ddsm/MidNet_3_0_demo.ipynb at master · aryamtos/midnet-cnn-ddsm](https://github.com/aryamtos/midnet-cnn-ddsm/blob/master/src/MidNet_3_0_demo.ipynb)

## Image Processing

## CBIS DDSM

This CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the Digital Database for Screening Mammography (DDSM) .The DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information.

[CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)

Tasks include displays, basic manipulations like cropping, flipping, rotating, segmentation etc.

[https://github.com/aryamtos/midnet-cnn-ddsm/tree/master/src/processing_augmentation_demo](https://github.com/aryamtos/midnet-cnn-ddsm/tree/master/src/processing_augmentation_demo)

1. Convert DICOM to PNG (DICOM is the standard for the communication of medical imaging information)
2. Extract image from directory
3. Segmentation and supress artifacts
4. Extraction pectoral Muscle
5. Extraction ROI

![Untitled](MidNet%20Convolutional%20Network/Untitled%201.png)

## Python Image Manipulation Tools

- Numpy
- PIL
- OpenCV

### Extraction Pectoral Muscle

[midnet-cnn-ddsm/pectoral_extraction_demo.ipynb at master · aryamtos/midnet-cnn-ddsm](https://github.com/aryamtos/midnet-cnn-ddsm/blob/master/src/pectoral_extraction_demo.ipynb)

1. Limiar Threshold
2.  Watershed Segmentation

![Untitled](MidNet%20Convolutional%20Network/Untitled%202.png)

## Data Augmentation

[midnet-cnn-ddsm/data_augmentation_cbisddsm.ipynb at master · aryamtos/midnet-cnn-ddsm](https://github.com/aryamtos/midnet-cnn-ddsm/blob/master/src/processing_augmentation_demo/data_augmentation_cbisddsm.ipynb)

Technique used to expand or enlarge your dataset by using the existing data of the dataset.

1. Rotation
2. Width Shifting
3. Height Shifting
4. Brightness
5. Shear Intensity
6. Rotation
7. Zoom
8. Channel Shift
9. Horizontal Flip
10. Vertical Flip

Link : [https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en](https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en)

### Paper

### Cite:

**[MATOS, A. N.](http://lattes.cnpq.br/4915122145392923)**; [AMBRÓSIO, Paulo E](http://lattes.cnpq.br/5034444360451621). Detecção de anormalidades em imagens mamográficas com deep learning. Brazilian Journal of Health Review, 2022.
