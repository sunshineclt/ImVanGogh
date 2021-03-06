import logging
import os
import time

import numpy as np
import tensorflow as tf

import evaluate
import transform
import utils
import vgg

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def compute_gram(features):
    features = np.reshape(features, (-1, features.shape[3]))
    return np.matmul(features.T, features) / features.size


def compute_gram_with_batch(layer):
    bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
    size = height * width * filters
    feats = tf.reshape(layer, (bs, height * width, filters))
    feats_transpose = tf.transpose(feats, perm=[0, 2, 1])
    grams = tf.matmul(feats_transpose, feats) / size
    return grams


def train(content_targets, style_target, content_weight, style_weight,
          tv_weight, vgg_path, epochs=2, checkpoint_frequency=1000,
          batch_size=4, save_dir='saver/fns.ckpt',
          learning_rate=1e-3, test_image=None, test_dir=None):

    batch_shape = (batch_size, 256, 256, 3)
    style_features = {}

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess, tf.variable_scope('VGG_style_target'):
        style_input = tf.placeholder(tf.float32, shape=(1,) + style_target.shape, name='style_input')
        style_input_pre = vgg.centralize(style_input)
        net = vgg.net(vgg_path, style_input_pre)
        style_image_input = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = sess.run(net[layer], feed_dict={style_input: style_image_input})
            style_features[layer] = compute_gram(features)

    with tf.Graph().as_default(), tf.Session() as sess:
        initial_image = tf.placeholder(tf.float32, shape=batch_shape, name="initial_image")
        initial_image_pre = vgg.centralize(initial_image)

        # transform initial image through transform net
        with tf.variable_scope("Transform_net"):
            transformed_image = transform.net(initial_image / 255.0)
            transformed_image_pre = vgg.centralize(transformed_image)

        # compute initial image's and transformed image's content features
        with tf.variable_scope("VGG", reuse=True):
            content_net = vgg.net(vgg_path, initial_image_pre)
            transformed_vgg_result = vgg.net(vgg_path, transformed_image_pre)

        assert utils.tensor_size_without_batch(content_net[CONTENT_LAYER]) == \
               utils.tensor_size_without_batch(transformed_vgg_result[CONTENT_LAYER])

        # compute content loss
        with tf.variable_scope("content_loss"):
            content_size = utils.tensor_size_without_batch(content_net[CONTENT_LAYER]) * batch_size
            content_loss = content_weight * 2 * \
                           tf.nn.l2_loss(content_net[CONTENT_LAYER] -
                                         transformed_vgg_result[CONTENT_LAYER]) \
                           / content_size

        # compute style loss
        with tf.variable_scope("style_loss"):
            style_losses = []
            for style_layer_name in STYLE_LAYERS:
                layer = transformed_vgg_result[style_layer_name]
                transformed_image_grams = compute_gram_with_batch(layer)
                style_gram = style_features[style_layer_name]
                style_losses.append(2 * tf.nn.l2_loss(transformed_image_grams - style_gram) / style_gram.size)
            style_losses = tf.stack(style_losses)
            style_loss = style_weight * tf.reduce_mean(style_losses)

        # compute variation denoising loss
        with tf.variable_scope("variation_loss"):
            tv_x_size = utils.tensor_size_without_batch(transformed_image[:, :, 1:, :])
            tv_y_size = utils.tensor_size_without_batch(transformed_image[:, 1:, :, :])
            x_tv = tf.nn.l2_loss(transformed_image[:, :, 1:, :] - transformed_image[:, :, :batch_shape[2] - 1, :])
            y_tv = tf.nn.l2_loss(transformed_image[:, 1:, :, :] - transformed_image[:, :batch_shape[1] - 1, :, :])
            tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size)

        with tf.variable_scope("total_loss"):
            loss = content_loss + style_loss + tv_loss

        with tf.variable_scope("train"):
            train_operator = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        runlog_dir = os.path.join(save_dir, 'run_log')
        train_writer = tf.summary.FileWriter(runlog_dir)
        train_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            num_train_data = len(content_targets)
            for iteration in range(num_train_data // batch_size):
                start_time = time.time()
                batch_data = np.zeros(batch_shape, dtype=np.float32)
                for i, image_name in enumerate(content_targets[iteration * batch_size: (iteration + 1) * batch_size]):
                    batch_data[i] = utils.get_image(image_name, (256, 256, 3)).astype(np.float32)

                assert batch_data.shape[0] == batch_size

                sess.run(train_operator, feed_dict={initial_image: batch_data})
                logging.debug("batch time: " + str(time.time() - start_time))

                if iteration % checkpoint_frequency == 0:
                    style_loss_record, content_loss_record, tv_loss_record, total_loss_record = \
                        sess.run([style_loss, content_loss, tv_loss, loss],
                                 feed_dict={
                                     initial_image: batch_data
                                 })
                    saver = tf.train.Saver()
                    checkpoint_dir = os.path.join(save_dir, 'checkpoint/style.ckpt')
                    saver.save(sess, checkpoint_dir)
                    logging.info('Epoch %d, Iteration: %d, Loss: %s' % (epoch, iteration, total_loss_record))
                    logging.info(
                        'style: %s, content:%s, tv: %s' % (style_loss_record, content_loss_record, tv_loss_record))
                    if test_image:
                        output_path = '%s/%s_%s.png' % (test_dir, epoch, iteration)
                        ckpt_dir = os.path.dirname(checkpoint_dir)
                        evaluate.transform_forward(test_image, output_path,
                                                   ckpt_dir)

        logging.info("Training complete. ")
        logging.info("Now time: %s" % (time.asctime(time.localtime(time.time()))))


def main():
    parser = utils.build_parser()
    args = parser.parse_args()
    utils.check_opts(args)
    logging.basicConfig(level=logging.INFO)

    style_target = utils.get_image(args.style_image)
    content_targets = utils.get_files(args.train_path)
    content_targets = utils.trim_data(content_targets, args.batch_size)

    train(content_targets, style_target, args.content_weight, args.style_weight, args.tv_weight,
          args.vgg_path, args.epochs, args.checkpoint_freq, args.batch_size,
          args.save_dir, args.learning_rate, args.test_image, args.test_dir)


if __name__ == '__main__':
    main()
