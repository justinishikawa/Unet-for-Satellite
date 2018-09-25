# Unet
This project focuses on using a modified Unet model for image classification of satellite images.  The original data for this project is from the [DSTL Kaggle Challenge](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)
The data for this project has been modified to only use the 8 band Multispectral Tiff collection and the classes have been reduced to 5: 
* Buildings
* Roads and Tracks
* Trees
* Crops
* Water

The model was trained on 24 images, each with 8 bands and 5 classification masks.  Image patches were augmented using ndimage and matrix math.  

Training and prediction was completed on a NVIDIA P100 GPU with 120GB of RAM

Libraries:  Keras, Tensorflow, Tifffile, Python3, numpy, scipy
