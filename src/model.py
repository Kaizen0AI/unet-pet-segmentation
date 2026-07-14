import tensorflow as tf
from src.blocks import encoder_block, decoder_block, bottleneck

def build_unet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    e1, p1 = encoder_block(inputs, conv_filters=(16, 16))
    e2, p2 = encoder_block(p1, conv_filters=(32, 32))
    e3, p3 = encoder_block(p2, conv_filters=(64, 64))
    e4, p4 = encoder_block(p3, conv_filters=(128, 128))

    # Bottleneck
    b = bottleneck(p4, conv_filters=256)

    # Decoder
    d1 = decoder_block(b, skip_features=e4, up_filters=128, conv_filters=(128, 128))
    d2 = decoder_block(d1, skip_features=e3, up_filters=64, conv_filters=(64, 64))
    d3 = decoder_block(d2, skip_features=e2, up_filters=32, conv_filters=(32, 32))
    d4 = decoder_block(d3, skip_features=e1, up_filters=16, conv_filters=(16, 16))

    # Output layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model