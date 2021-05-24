from segmentation_models import UNET
from segmentation_utils import dataset_setup, conv2d_block
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


# Directories containing the train and val data
train_files = "../data/cityscapes_data/train/"
test_files = "../data/cityscapes_data/val/"

# Setup image-label pairs
x, y = dataset_setup(data_dir=train_files, n_ims=2975, offset_bias=0, img_dim=256)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_test, y_test = dataset_setup(data_dir=test_files, n_ims=500, offset_bias=0, img_dim=256)

# Initialize UNET
model = UNET(input_shape=(256,256,3), conv_block=conv2d_block, n_filters=32, dropout=0.5, padding='same', batch_norm=True)
print(model.summary())

# Compile model with specified optimizer and loss
model.compile(optimizer='adam', loss='MSE')

# Track model history as it trains
h = model.fit(x_train, y_train, epochs=20, shuffle=True, batch_size=10, validation_data=(x_val, y_val))

# Print results for training MSE and validation MSE
plt.plot(h.history['loss'])
plt.show()
plt.plot(h.history['val_loss'])
plt.show()

#show the result
pp2 = model.predict(x_test[:,:,:,:], batch_size=1)

ni = 5
for k in range(ni):

    plt.figure(figsize=(10,30))
    plt.subplot(ni,3,1+k*3)
    plt.imshow(x_test[k])
    plt.subplot(ni,3,2+k*3)
    plt.imshow(y_test[k])
    plt.subplot(ni,3,3+k*3)
    plt.imshow(pp2[k])

intersection = np.logical_and(y_test[:,:,:,:], pp2)
union = np.logical_or(y_test[:,:,:,:], pp2)
iou_score = np.sum(intersection) / np.sum(union)
print(iou_score)

diff = y_test[1] - pp2[1] 
m_norm = np.sum(abs(diff))  
print(m_norm)
