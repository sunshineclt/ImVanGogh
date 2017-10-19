import tensorflow as tf
import numpy as np
import scipy.io

MEAN_PIXEL = np.array([123.68, 116.779, 103.939])


def net(vgg_pretrained_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(vgg_pretrained_path)
    # mean = data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    network_layers = {}
    out = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # Because matconvnet's weights are [width, height, in_channels, out_channels]
            # while   tensorflow's weights are [height, width, in_channels, out_channels]
            # So we need to transpose
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            out = _conv_layer(out, kernels, bias)
        elif kind == 'relu':
            out = tf.nn.relu(out)
        elif kind == 'pool':
            out = _pool_layer(out)
        network_layers[name] = out

    assert len(network_layers) == len(layers)
    return network_layers


def _conv_layer(input_tensor, weights, bias):
    conv = tf.nn.conv2d(input=input_tensor,
                        filter=tf.constant(weights),
                        strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input_tensor):
    return tf.nn.max_pool(input_tensor,
                          ksize=(1, 2, 2, 1),
                          strides=(1, 2, 2, 1),
                          padding='SAME')


def centralize(image):
    return image - MEAN_PIXEL


def decentralize(image):
    return image + MEAN_PIXEL
