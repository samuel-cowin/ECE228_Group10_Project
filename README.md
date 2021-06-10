# Group 10 project folder for ECE228 Machine Learning for Physical Applications with Professor Peter Gerstoft
### Enclosing Semantic Segmentation and Computer Vision for Autonomous Vehicles
### Dataset chosen is Cityscapes image-pairs from Kaggle as seen here https://www.kaggle.com/dansbecker/cityscapes-image-pairs and included in the data folder
### UNET folder contains an implementation of UNET/UNET++ for performance comparison
### SegNet folder contains an encoder-decoder implementation without max indices tracking
### Pre-Trained folder contains code from another repository (refernce within folder) for implementing and training FCN
<br><br>
## Example of running code from cloned repo outlined below - package requirements in requirements.txt
### Necessary to have high-RAM device (>12 GB) for SegNet .ipynb and U-net .py files
### For U-net/U-net++, there are .py files or .ipynb files to be used. 
<br>-.py files - run UNET.py within the UNET/folder. This is an example of running with color pixels as the reconstructed output
<br>-.ipynb files - use Google Colab with data saved on Google Drive, mount to drive, and change path to data. Run other cells as implemented. Outputs class labels as reconstructed output.
### For SegNet, .ipynb files need to be used. 
<br>-.ipynb files - use Google Colab with data saved on Google Drive, mount to drive, and change path to data. Run other cells as implemented. Outputs class labels as reconstructed output.
