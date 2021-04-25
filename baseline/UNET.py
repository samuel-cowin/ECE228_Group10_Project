from segmentation_models import UNET
from segmentation_utils import dataset_setup, conv2d_block
import matplotlib.pyplot as plt

# Directories containing the train and val data
train_files = "../data/cityscapes_data/train/"
val_files = "../data/cityscapes_data/val/"

# Setup image-label pairs
x_train, y_train = dataset_setup(data_dir=train_files, n_ims=2975, offset_bias=0, img_dim=256)
x_val, y_val = dataset_setup(data_dir=val_files, n_ims=500, offset_bias=0, img_dim=256)

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