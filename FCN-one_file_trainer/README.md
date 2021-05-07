### To run this model:

#### pre-trained model installation:
FCN : https://drive.google.com/file/d/1WElbk7ogK3e3-yEDP0yXfy4sCpbYL4yP/view?usp=sharing
put this file in model directory


#### Data Processing
The first few blocks of code in run_one_file_predict.ipynb processes the data, separating the ground truth and save them in the directory. 

#### Run
run_one_file_predict.ipynb contains the code to execute the training. Now I only made it to train two images, since it takes some time to train each(about 5 minutes per image on datahub with 32GB RAM). I tried to train 3 images and the kernel died. 

#### Reference

Original github repository :https://github.com/hellochick/semantic-segmentation-tensorflow
