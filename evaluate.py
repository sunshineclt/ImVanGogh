from __future__ import print_function

import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

import transform
from utils import save_img, get_image


def transform_forward(path_in, path_out, checkpoint_dir):
    image_input = get_image(path_in)
    image_input = np.array([image_input])
    image_shape = image_input.shape

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session(config=config) as sess:
        initial_image = tf.placeholder(tf.float32,
                                       shape=image_shape,
                                       name='initial_image')
        with tf.variable_scope("Transform_net"):
            transformed_image_tensor = transform.net(initial_image)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        transformed_image = sess.run(transformed_image_tensor,
                                     feed_dict={initial_image: image_input})

        save_img(path_out, transformed_image[0])


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='dir or file to transform',
                        metavar='IN_PATH', required=True)
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help="'destination (dir or file) of transformed file or files'",
                        metavar='OUT_PATH', required=True)
    return parser


def check_opts(args):
    assert os.path.exists(args.checkpoint_dir), "checkpoint dir not found!"
    assert os.path.exists(args.in_path), "in path dir not found!"
    if os.path.isdir(args.out_path):
        assert os.path.exists(args.out_path), 'out dir not found!'


def main():
    parser = build_parser()
    args = parser.parse_args()
    check_opts(args)

    if os.path.exists(args.out_path) and os.path.isdir(args.out_path):
        out_path = os.path.join(args.out_path, os.path.basename(args.in_path))
    else:
        out_path = args.out_path
    transform_forward(args.in_path, out_path, args.checkpoint_dir)


if __name__ == '__main__':
    main()
