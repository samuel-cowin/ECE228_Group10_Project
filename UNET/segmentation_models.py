from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2DTranspose, Conv2D, concatenate
from tensorflow.keras import Model, Input

from segmentation_utils import conv2d_block


def UNET(input_shape=(256,256,3), conv_block=conv2d_block, n_filters=32, dropout=0.5, padding='same', batch_norm=True):
    """
    UNET architecture as originally outlined in https://arxiv.org/pdf/1505.04597.pdf with modifications 
    to fit different input dimensions. 

    Inputs
    --
    input_shape: tuple(int)
        Tuple in 3D corresponding to the dimensions of the input images
    conv_block: func
        Custom block method to perform consecutive convolutions with optional batch normalization
    n_filters: int
        Number of filters corresponding to depth of input for next layer
    dropout: float
        Dropout percentage hyperparameter to tune overfitting
    padding: string
        Descriptor determining if padding maintain size during convolutions
    batch_norm: bool
        Determines if batch normalization is used

    Outputs
    --
    model: Model
        Returns model architecture without compile
    """

    tensor = Input(shape=input_shape)

    print('Contracting Path')
    c1 = conv_block(tensor, n_filters * 1, filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv_block(p1, n_filters * 2, filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv_block(p2, n_filters * 4,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv_block(p3, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv_block(p4, n_filters * 16, filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)

    print('Expanding Path')
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv_block(u6, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding=padding)(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv_block(u7, n_filters * 4,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding=padding)(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv_block(u8, n_filters * 2,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding=padding)(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv_block(u9, n_filters * 1,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[tensor], outputs=[outputs])

    # Return model architecture
    return model


def UNET_plusplus(input_shape=(128,128,3), conv_block=conv2d_block, n_filters=32, dropout=0.5, padding='same', batch_norm=True):
    """
    UNET++ architecture as originally outlined in https://arxiv.org/pdf/1807.10165.pdf with modifications 
    to fit different input dimensions. 

    Inputs
    --
    input_shape: tuple(int)
        Tuple in 3D corresponding to the dimensions of the input images
    conv_block: func
        Custom block method to perform consecutive convolutions with optional batch normalization
    n_filters: int
        Number of filters corresponding to depth of input for next layer
    dropout: float
        Dropout percentage hyperparameter to tune overfitting
    padding: string
        Descriptor determining if padding maintain size during convolutions
    batch_norm: bool
        Determines if batch normalization is used

    Outputs
    --
    model: Model
        Returns model architecture without compile
    """

    tensor = Input(shape=input_shape)

    print('Contracting Path')
    
    # First Diagonal
    c00 = conv_block(tensor, n_filters * 1, filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p00 = MaxPooling2D((2, 2))(c00)
    p00 = Dropout(dropout)(p00)
    
    c10 = conv_block(p00, n_filters * 2, filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p10 = MaxPooling2D((2, 2))(c10)
    p10 = Dropout(dropout)(p10)

    c20 = conv_block(p10, n_filters * 4,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p20 = MaxPooling2D((2, 2))(c20)
    p20 = Dropout(dropout)(p20)

    c30 = conv_block(p20, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    p30 = MaxPooling2D((2, 2))(c30)
    p30 = Dropout(dropout)(p30)

    c40 = conv_block(p30, n_filters * 16, filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    #Second Diagonal
    
    u01 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c10)
    u01 = concatenate([u01, c00])
    u01 = Dropout(dropout)(u01)
    c01 = conv_block(u01, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    u11 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c20)
    u11 = concatenate([u11, c10])
    u11 = Dropout(dropout)(u11)
    c11 = conv_block(u11, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    u21 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c30)
    u21 = concatenate([u21, c20])
    u21 = Dropout(dropout)(u21)
    c21 = conv_block(u21, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)   
    
    u31 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c40)
    u31 = concatenate([u31, c30])
    u31 = Dropout(dropout)(u31)
    c31 = conv_block(u31, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)    
    
    #Third Diagonal
    
    u02 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c11)
    u02 = concatenate([u02, c01])
    u02 = Dropout(dropout)(u02)
    c02 = conv_block(u02, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)        

    u12 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c21)
    u12 = concatenate([u12, c11])
    u12 = Dropout(dropout)(u12)
    c12 = conv_block(u12, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    u22 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c31)
    u22 = concatenate([u22, c21])
    u22 = Dropout(dropout)(u22)
    c22 = conv_block(u22, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    #Fourth Diagonal
    
    u03 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c12)
    u03 = concatenate([u03, c02])
    u03 = Dropout(dropout)(u03)
    c03 = conv_block(u03, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    u13 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c22)
    u13 = concatenate([u13, c12])
    u13 = Dropout(dropout)(u13)
    c13 = conv_block(u13, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)
    
    #Fifth Diagonal
    
    u04 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding=padding)(c13)
    u04 = concatenate([u04, c03])
    u04 = Dropout(dropout)(u04)
    c04 = conv_block(u04, n_filters * 8,  filter_size=3, activation='relu', pad=padding, batch_norm=batch_norm)  

    #Outputs
    
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c04)
    model = Model(inputs=[tensor], outputs=[outputs])

    # Return model architecture
    return model