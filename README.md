# Group 10 project folder for ECE228 Machine Learning for Physical Applications with Professor Peter Gerstoft
### Enclosing Semantic Segmentation and Computer Vision for Autonomous Vehicles
### Dataset chosen is Cityscapes image-pairs from Kaggle as seen here https://www.kaggle.com/dansbecker/cityscapes-image-pairs and included in the data folder
### UNET folder contains an implementation of UNET/UNET++ for performance comparison
### SegNet folder contains an encoder-decoder implementation without max indices tracking
### Pre-Trained folder contains code from another repository (refernce within folder) for implementing and training FCN
<br>

## Example of running code from cloned repo outlined below
### Requirements:
Use python 3.8+ <br /><br />
seaborn==0.11.1
<br />opencv_contrib_python==4.4.0.44
<br />Keras==2.4.3
<br />matplotlib==3.2.2
<br />numpy==1.19.5
<br />ipython==7.24.1
<br />Pillow==8.3.2
<br />protobuf==3.17.3
<br />scikit_learn==0.24.2
<br />tensorflow==2.5.1
### Necessary to have high-RAM device (>12 GB) for SegNet .ipynb and U-net .py files
### For U-net/U-net++, there are .py files or .ipynb files to be used. 
-.py files - run UNET.py within the UNET/folder. This is an example of running with color pixels as the reconstructed output
<br>-.ipynb files - use Google Colab with data saved on Google Drive, mount to drive, and change path to data. Run other cells as implemented. Outputs class labels as reconstructed output.
### For SegNet, .ipynb files need to be used. 
-.ipynb files - use Google Colab with data saved on Google Drive, mount to drive, and change path to data. Run other cells as implemented. This is an example of running with color pixels as the reconstructed output
<br><br>
## Example of using U-net/U-net++ from .py files
Import the model architecture<br />
```python
from segmentation_models import UNET
```
Initialize the model with certain hyperparameters<br />
```python
model = UNET(input_shape=(256,256,3), conv_block=conv2d_block, n_filters=32, dropout=0.5, padding='same', batch_norm=True)
```
From here, data extraction and model compilation/training are project dependent. Full runnable example is included in the UNET folder.
