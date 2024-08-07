# Prediction of mouse seminiferous tubule stage

This is the code for [Deep learning based automated prediction of mouse seminiferous tubule stage by using bright-field microscopy]().
This project is carried out in [Funahashi Lab. at Keio University](https://fun.bio.keio.ac.jp/).

## Overview


## Performance


## Requirements

- [Python 3.7.6+](https://www.python.org/downloads/)
- [Pytorch 1.13.1](https://pytorch.org/)
- [Torchvision 0.14.1](https://pytorch.org/vision/stable/index.html)
- [Matplotlib 3.5.3](https://matplotlib.org/)
- [NumPy 1.21.6](http://www.numpy.org)
- [scikit-image 0.16.1](http://scikit-image.org/)

See ```requirements.txt``` for details. 


## QuickStart

1. Download this repository by `git clone`.
   ```sh
   % git clone git@github.com:funalab/seminiferous_tubule_stage_classification.git
   ```
2. Install requirements.
   ```sh
   % cd seminiferous_tubule_stage_classification/
   % python -m venv venv
   % source ./venv/bin/activate
   % pip install --upgrade pip
   % pip install -r requirements.txt
   ```
4. Download datasets and learned model.
   - On Linux:

      ```sh
      % wget -O models/ "XXX"
      % wget -O seminiferous_tubule_dataset.tar.gz "XXX"
      % tar zxvf seminiferous_tubule_dataset.tar.gz
      ```

   - on macOS:
     ```sh
     % curl --output models/ "XXX"
     % curl --output seminiferous_tubule_dataset.tar.gz "XXX"
     % tar zxvf seminiferous_tubule_dataset.tar.gz
     ```
6. Run the model.
    - On CPU:

        ```sh

        % python src/tools/test.py --conf_file confs/test_resnet.cfg
        ```

    - On CPU (Negative value of GPU ID indicates CPU)):

        ```sh
        % python src/tools/test.py
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
