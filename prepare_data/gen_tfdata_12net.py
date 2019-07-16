"""Generate tfrecords file for pnet, which has input size of 12*12*3."""

import sys
import random

import cv2
import tensorflow as tf
import numpy as np
import numpy.random as npr

from tools import view_bar, bytes_feature

sys.path.append('../')


def main():

    size = 12
    net = 'native_'+str(size)

    with open('%s/pos_%s.txt' % (net, size), 'r') as f:
        pos = f.readlines()
    with open('%s/neg_%s.txt' % (net, size), 'r') as f:
        neg = f.readlines()
    with open('%s/part_%s.txt' % (net, size), 'r') as f:
        part = f.readlines()

    print('\n'+'pos')
    filename_cls = 'pnet_data_for_cls.tfrecords'
    print('Writing')
    examples = []
    writer = tf.python_io.TFRecordWriter(filename_cls)
    cur_ = 0
    sum_ = len(pos)
    for line in pos:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([0, 1], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)

    print('\n'+'neg')
    cur_ = 0
    neg_keep = npr.choice(len(neg), size=1000000, replace=False)
    sum_ = len(neg_keep)
    for i in neg_keep:
        line = neg[i]
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([1, 0], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
    print(len(examples))
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()

    print('\n'+'pos')
    cur_ = 0
    filename_roi = 'pnet_data_for_bbx.tfrecords'
    print('Writing')
    sum_ = len(pos)
    examples = []
    writer = tf.python_io.TFRecordWriter(filename_roi)
    for line in pos:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([float(words[2]), float(words[3]),
                          float(words[4]), float(words[5])],
                         dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)

    print('\n'+'part')
    cur_ = 0
    part_keep = npr.choice(len(part), size=300000, replace=False)
    sum_ = len(part_keep)
    for i in part_keep:
        view_bar(cur_, sum_)
        line = part[i]
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 12 or w != 12:
            im = cv2.resize(im, (12, 12))
        im = im.astype('uint8')
        label = np.array([float(words[2]), float(words[3]),
                          float(words[4]), float(words[5])],
                         dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
    print(len(examples))
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
