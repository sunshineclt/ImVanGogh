import scipy.misc
import numpy as np
from argparse import ArgumentParser
import os
import logging
import functools


def get_image(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')  # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size:
        img = scipy.misc.imresize(img, img_size)
    return img


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


def get_files(directory):
    files = []
    for (_, _, filenames) in os.walk(directory):
        files.extend(filenames)
        break
    return [os.path.join(directory, x) for x in files]


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--style-image', type=str,
                        dest='style_image', help='style image path',
                        required=True)
    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        default='/big/dataset/train2014')
    parser.add_argument('--test-image', type=str,
                        dest='test_image', help='test image path (in train process transform test image)',
                        default=False)
    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        default='/big/run_log/middle_result')
    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        default='/big/dataset/imagenet-vgg-verydeep-19.mat')
    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        default=2)
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        default=10)
    parser.add_argument('--checkpoint-freq', type=int,
                        dest='checkpoint_freq', help='checkpoint frequency',
                        default=2000)
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        default='/big/run_log/checkpoint')
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        default=7.5)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        default=100)
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        default=200)
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        default=1e-3)
    return parser


def check_opts(args):
    assert os.path.exists(args.checkpoint_dir), "checkpoint dir not found!"
    assert os.path.exists(args.style_image), "style image not found!"
    assert os.path.exists(args.train_path), "train dir not found!"
    assert os.path.exists(args.vgg_path), "vgg parameter data not found!"
    if args.test or args.test_dir:
        assert os.path.exists(args.test_image), "test img not found!"
        assert os.path.exists(args.test_dir), "test dir not found!"
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.checkpoint_iterations > 0
    assert args.content_weight >= 0
    assert args.style_weight >= 0
    assert args.tv_weight >= 0
    assert args.learning_rate >= 0


def trim_data(content_targets, batch_size):
    mod = len(content_targets) % batch_size
    if mod > 0:
        logging.INFO("Train set has been trimmed slightly..")
        return content_targets[:-mod]


def tensor_size_without_batch(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


class TensorFlowFunction(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        outputs_list = tf.get_default_session().run(self._outputs, feeds)[:len(self._outputs)]
        return outputs_list
