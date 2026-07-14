import tensorflow as tf

def encoder_block(inputs, conv_filters, kernel_size=(3,3), strides=1):
    """
    Create an encoder block with convolutional and pooling layers.
    
    Parameters
    ----------
    inputs:
        Input tensor.
    conv_filters:
        Tuple (f1, f2) specifying the number of filters
        used by the two convolution layers.
    """
    f1, f2 = conv_filters
    x = tf.keras.layers.Conv2D(f1, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f2, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    p = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    return x, p

def decoder_block(inputs, skip_features, up_filters, conv_filters, kernel_size=(3,3), strides=1):
    """
    Create a decoder block with upsampling and convolutional layers.

    Parameters
    ----------
    inputs:
        Input tensor from the previous decoder stage.
    skip_features:
        Corresponding encoder feature map.
    up_filters:
        Number of filters for the upsampling layer.
    conv_filters:
        Tuple (f1, f2) specifying the number of filters
        used by the two convolution layers after concatenation.
    """
    f1 = up_filters
    f2, f3 = conv_filters
    x = tf.keras.layers.Conv2DTranspose(f1, kernel_size, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = tf.keras.layers.Conv2D(f2, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f3, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def bottleneck(inputs, conv_filters, kernel_size=(3,3), strides=1):
    """
    Create a bottleneck block with convolutional layers.
    """
    x = tf.keras.layers.Conv2D(conv_filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(conv_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x
