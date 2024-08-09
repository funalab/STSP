# STSP: Seminiferous Tubule Stage Prediction

[![biorXiv Paper](https://img.shields.io/badge/DOI-XXX-blue)]()

This is the code for our paper [Deep learning based automated prediction of mouse seminiferous tubule stage by using bright-field microscopy](). 
This project is carried out in [Funahashi Lab. at Keio University](https://fun.bio.keio.ac.jp/).


## Overview

Our model performs to predict the stage from bright-field microscope images of mouse seminiferous tubules, in which there are 12 developmental stages.
The maximum prediction accuracy of our model is 79.58%, which increases to 98.33% when a prediction error of ±1 stage is allowed.

![overview](figs/overview.jpg)


## Requirements

- [Python 3.7.6+](https://www.python.org/downloads/)
- [Pytorch 1.9.0](https://pytorch.org/)
- [Torchvision 0.10.0](https://pytorch.org/vision/stable/index.html)
- [Matplotlib 3.5.3](https://matplotlib.org/)
- [NumPy 1.21.6](http://www.numpy.org)
- [scikit-image 0.16.1](http://scikit-image.org/)

See ```requirements.txt``` for details. 


## QuickStart

1. Download this repository by `git clone`.
   ```sh
   % git clone git@github.com:funalab/STSP.git
   ```
2. Install requirements.
   ```sh
   % cd STSP/
   % python -m venv venv
   % source ./venv/bin/activate
   % pip install --upgrade pip
   % pip install -r requirements.txt
   ```
3. Download datasets and learned model.
   - On Linux:

      ```sh
      % wget -O models.tar.gz "XXX"
      % tar zxvf models.tar.gz
      % wget -O datasets.tar.gz "XXX"
      % tar zxvf datasets.tar.gz
      ```

   - on macOS:
     ```sh
     % curl --output models.tar.gz "XXX"
     % tar zxvf models.tar.gz
     % curl --output datasets.tar.gz "XXX"
     % tar zxvf datasets.tar.gz
     ```
4. Run the model.

    The following command will run the test on the GPU (`device=cuda:0`).
    If you want to run on CPU, open `confs/test_resnet.cfg` and rewrite `device=cpu`.
    ```sh
    % python src/tools/test.py --conf_file confs/test_resnet.cfg
    ```


## How to train and run model with your data

1. At first, prepare the dataset following the directory structure as follows:

    ```
    your_dataset/
           +-- images/  (train, validation, and test images)
           |       +-- image_1-1.tif  (stage 1)
           |       +-- image_1-2.tif  (stage 1)
           |       +-- image_2-1.tif  (stage 2)
           |       |         :        
           |       |         :        
           |       +-- image_12-14.tif  (stage 12)
           |       +-- image_12-15.tif  (stage 12)
           | 
           +-- split_list/
           |           +-- train.txt   (List of file names for train images)
           |           +-- test.txt    (List of file names for test images)
           +-- label_list.txt
    ```


2. Train model with the above-prepared dataset.

3. Run model with the above-prepared dataset.


## Acknowledgements

This research was funded by JST CREST Grant Number JPMJCR21N1 to M.I. and A.F. The NVIDIA Tesla V100 was used in the miniRAIDEN computer server owned by RIKEN.
