import tensorflow as tf

WEIGHTS_INIT_STDEV = .1


def net(image):
    conv1 = _conv_layer(image, 32, 9, 1, "conv1")
    conv2 = _conv_layer(conv1, 64, 3, 2, "conv2")
    conv3 = _conv_layer(conv2, 128, 3, 2, "conv3")
    resid1 = _residual_block(conv3, 3, "resid1")
    resid2 = _residual_block(resid1, 3, "resid2")
    resid3 = _residual_block(resid2, 3, "resid3")
    resid4 = _residual_block(resid3, 3, "resid4")
    resid5 = _residual_block(resid4, 3, "resid5")
    conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2, "transconv1")
    conv_t2 = _conv_transpose_layer(conv_t1, 32, 3, 2, "transconv2")
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, "conv4", relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


def _conv_layer(input_layer, num_filters, filter_size, strides, scope, relu=True):
    with tf.VariableScope(scope):
        weights_init = _create_conv_parameters(input_layer, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        output_net = tf.nn.conv2d(input_layer, weights_init, strides_shape, padding='SAME')
        output_net = _instance_norm(output_net)
        if relu:
            output_net = tf.nn.relu(output_net)
        return output_net


def _conv_transpose_layer(input_layer, num_filters, filter_size, strides, scope):
    with tf.VariableScope(scope):
        weights_init = _create_conv_parameters(input_layer, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i.value for i in input_layer.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        new_shape = tf.stack([batch_size, new_rows, new_cols, num_filters])
        strides_shape = [1, strides, strides, 1]

        input_layer = tf.nn.conv2d_transpose(input_layer, weights_init, new_shape, strides_shape, padding='SAME')
        input_layer = _instance_norm(input_layer)
        return tf.nn.relu(input_layer)


def _residual_block(input_layer, filter_size, scope):
    with tf.VariableScope(scope):
        tmp = _conv_layer(input_layer, 128, filter_size, 1, "conv1")
        return input_layer + _conv_layer(tmp, 128, filter_size, 1, "conv2", relu=False)


def _instance_norm(input_layer):
    # batch, rows, cols, channels = [i.value for i in input_layer.get_shape()]
    # var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(input_layer, [1, 2], keep_dims=True)
    # shift = tf.Variable(tf.zeros(var_shape))
    # scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (input_layer - mu) / (sigma_sq + epsilon) ** .5
    # return scale * normalized + shift
    return normalized


def _create_conv_parameters(input_layer, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in input_layer.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights
