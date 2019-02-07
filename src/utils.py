# TODO: image processing은 tensorflow으로...?
from __future__ import division

import os
import json
import collections
import pathlib
import tensorflow as tf


class ParsingJson:

    def __init__(self, filename):
        """

        :param filename:
        """

        self.filename = filename

    def parsing(self):
        if not self._isEmpty():
            raise ValueError('Not Found File')

        with open(self.filename) as f:
            architecture = json.load(f,
                                     object_pairs_hook=collections.OrderedDict)

        return architecture

    def _isEmpty(self):
        """
        check existence of file

        :return:
        """

        return os.path.exists(self.filename)

    def specification(self):
        # TODO: count the number of parameter, print architecture
        pass


def getfile(filename, url, dir_to_save):
    path_to_download = os.path.abspath('../' + dir_to_save)

    data_root = tf.keras.utils.get_file(path_to_download + filename,
                                        url,
                                        untar=True,
                                        cache_dir=path_to_download,
                                        cache_subdir=path_to_download)

    print('Success to download!')
    return pathlib.Path(data_root)


class ImageAugmentation:
    pass


class ImageProcessing:
    """
    Image processing with tensorflow
    """
    def __init__(self,
                 filename,
                 url,
                 channel=3,
                 dir_to_save='images'):

        self.label_to_index = None
        self.all_image_paths = None
        self.all_image_labels = None
        self.channel = channel
        self.height = None
        self.width = None
        self.dataRoot = getfile(filename, url, dir_to_save)

    def getAllImagePaths(self):

        all_image_paths = list(self.dataRoot.glob('*/*'))
        self.all_image_paths = [str(path)
                                for path in all_image_paths
                                if 'Thumbs.db' not in str(path)]

    def labeling2images(self):
        label_names = sorted(item.name for item in self.dataRoot.glob('*/') if item.is_dir())

        self.label_to_index = dict((name, index) for index, name in enumerate(label_names))

        self.all_image_labels = [self.label_to_index[pathlib.Path(path).parent.name]
                                 for path in self.all_image_paths]

    def readImages(self, path, label):
        image = tf.read_file(path)

        return self.preprocessImage(image), label

    def preprocessImage(self, image):
        image = tf.image.decode_jpeg(image,
                                     channels=self.channel)
        image = tf.image \
            .resize_image_with_crop_or_pad(image, self.height, self.width)

        return tf.math.divide(tf.to_float(image), 255)  # normalize to [0,1] range

    def getDataset(self, height, width, batchSize):
        self.getAllImagePaths()
        self.labeling2images()

        self.height = height
        self.width = width

        path_ds = tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels))

        dataset = path_ds \
            .map(self.readImages, -1) \
            .shuffle(len(self.all_image_paths)) \
            .repeat() \
            .batch(batchSize) \
            .prefetch(1)

        return dataset


if __name__ == '__main__':
    # tf.enable_eager_execution()
    # test = ImageProcessing('/flower_photos',
    #                        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz')
    # dataset = test.getDataset(224, 224, 128)
    # image, label = next(iter(dataset))
    # print(image)
    # for image, label in dataset:
    #     print(image, label);break

    pass