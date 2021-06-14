import scipy.misc
import numpy as np
import os
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras import backend as K

class ImageData:

    def __init__(self, load_size, channels):
        self.load_size = load_size
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img


def load_mnist(size=64):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    # x = np.expand_dims(x, axis=-1)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])
    x = np.expand_dims(x, axis=-1)
    return x
##########################################################################################
# def load_cifar10(size) :
#     ##data_X1, data_y1 = load_cifar10()
#     (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
#     train_data = normalize(train_data)
#     test_data = normalize(test_data)
#
#     x = np.concatenate((train_data, test_data), axis=0)
#     # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)
#
#     seed = 777
#     np.random.seed(seed)
#     np.random.shuffle(x)
#     # np.random.seed(seed)
#     # np.random.shuffle(y)
#
#     x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])
#
#     return x
#####################################################################################
def load_cifar10():
    # data_dir = "D:\test11111111111111/test11111111111111/cifar-10-batches-py"
    #
    # #filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
    # filenames = ['data_batch_%d.bin' % i for i in xrange(1, 6)]
    # filenames.append(os.path.join(data_dir, 'test_batch'))
    filename = 'cifar-10-batches-py'
    # origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = 'D:/test11111111111111/test11111111111111/cifar-10-batches-py'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    dataX = x_train
    labely = y_train
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(dataX)
    np.random.seed(seed)
    np.random.shuffle(labely)

    y_vec = np.zeros((len(labely), 10), dtype=np.float)
    for i, label in enumerate(labely):
        y_vec[i, labely[i]] = 1.0

    return dataX / 255., y_vec
#####################################################################################

def load_data(dataset_name, size=64) :
    if dataset_name == 'mnist' :
        x = load_mnist(size)
    elif dataset_name == 'cifar10' :
        x = load_cifar10(size)
    else :

        x = glob(os.path.join("./dataset", dataset_name, '*.*'))

    return x


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')