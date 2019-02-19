# TODO: test용 아주 작은 dataset 만들기. 20개 정도
import collections
import json
import multiprocessing as mp
import os
import pathlib
import random
import tempfile

import numpy as np
import PIL.Image
import tensorflow as tf
from sklearn.model_selection import train_test_split


def getfile(filename, url, dir_to_save):
    path_to_download = os.path.abspath('../' + dir_to_save)

    data_root = tf.keras \
        .utils.get_file(path_to_download + filename,
                        url,
                        untar=True,
                        cache_dir=path_to_download,
                        cache_subdir=path_to_download)

    print('Success to download!')

    return pathlib.Path(data_root)


class Byte2image:
    def __init__(self,
                 image_path,
                 label_path,
                 channel,
                 path):
        """

        :param image_path:
        :param label_path:
        :param channel:
        :param path: path to save images
        """
        self.image_path = image_path
        self.label_path = label_path
        self.channel = 'RGB' if channel > 1 else 'L'
        self.path = path

    def transform(self):

        _images, _labels = self.byte2np()
        self.makeDirectory(_labels)
        self.np2image(_images, _labels)

    def makeDirectory(self, labels):
        label_list = list(set(labels))
        for label in label_list:
            pathlib.Path(
                self.path+'/%s' % label) \
                .mkdir(parents=True,
                       exist_ok=True)

    def byte2np(self):
        intType = np.dtype('int32') \
            .newbyteorder('>')
        nMetaDataBytes = 4 * intType.itemsize

        images = np.fromfile(self.image_path,
                             dtype='ubyte')
        magicBytes, nImages, width, height = np.frombuffer(
            images[:nMetaDataBytes].tobytes(),
            intType)
        images = images[nMetaDataBytes:] \
            .astype(dtype='float32') \
            .reshape([nImages, width, height])
        labels = np.fromfile(self.label_path,
                             dtype='ubyte')[2 * intType.itemsize:]
        return images, labels

    def np2image(self, _images, _labels):
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)
        _paths = [tempfile.mktemp('.jpg',
                                  str(_label) + '/',
                                  self.path) for _label in _labels]
        pool.map(self._np2image,
                 zip(_images, _paths))

    def _np2image(self, arg):
        arr, path = arg
        PIL.Image \
            .fromarray(arr) \
            .convert(self.channel) \
            .save(path)


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


class ImagePipeline:
    """
    Image processing with tensorflow
    """
    # TODO: tfrecord 읽어 오기 혹은 raw image 사용
    def __init__(self,
                 width,
                 height,
                 channel,
                 filename,
                 url,
                 dataRoot=None,
                 dir_to_save='images'):

        self.width = width
        self.height = height
        self.channel = channel
        self.dataRoot = getfile(filename, url, dir_to_save)\
            if dataRoot is None else pathlib.Path(dataRoot)

        self.label_to_index = None
        self.all_image_labels = None
        self.labels = None

        self.all_image_paths = None
        self.label_to_index = None
        self.all_image_labels = None
        self.labels = None

        self._getAllImagePaths()
        self._labeling2images()

    def _getAllImagePaths(self):

        all_image_paths = list(self.dataRoot.glob('*/*'))
        self.all_image_paths = [str(path)
                                for path in all_image_paths
                                if 'Thumbs.db' not in str(path)]

    def _labeling2images(self):

        label_names = sorted(item.name for item in self.dataRoot.glob('*/') if item.is_dir())
        self.label_to_index = dict((name, index) for index, name in enumerate(label_names))
        self.labels = len(label_names)
        self.all_image_labels = [self.label_to_index[pathlib.Path(path).parent.name]
                                 for path in self.all_image_paths]

    def readImages(self, path, label):

        image = tf.image.decode_jpeg(
            tf.read_file(path),
            self.channel
        )
        image = tf.image.resize_images(image, [self.height, self.width])

        return self.demeanImage(image),\
               self.oneHotEncoding(label)

    def oneHotEncoding(self, label):

        return tf.one_hot(label, self.labels)

    def demeanImage(self, image):

        return image - self.getMeanImage(image)

    def getMeanImage(self, image):

        return tf.math \
            .reduce_mean(image,
                         axis=[0, 1])

    def getDataset(self, batchsize=128):
        """

        :param batchsize: batch size
        :return:
        """

        self._getAllImagePaths()
        self._labeling2images()

        train, test, train_label, test_label = train_test_split(
            self.all_image_paths,
            self.all_image_labels,
            test_size=0.33,
            stratify=self.all_image_labels
        )

        trainset = tf.data.Dataset \
            .from_tensor_slices((train,
                                 train_label))\
            .map(self.readImages, -1) \
            .shuffle(len(train)) \
            .repeat() \
            .batch(batchsize) \
            .prefetch(1)

        testset = tf.data.Dataset \
            .from_tensor_slices((test,
                                 test_label)) \
            .map(self.readImages, -1) \
            .shuffle(len(test)) \
            .repeat() \
            .batch(batchsize) \
            .prefetch(1)

        return trainset, testset


class ImageAugmentation(ImagePipeline):
    """
    augmentation, pre-processing of image
    변환은 다른 library 사용하고 변환된 이미지를 tfrecord로 저장
    아니면 tensorflow 이용하고 tfrecord 포기
    """
    # TODO: data augmentation 후 tfrecord로 저장
    # 크기조절(scaling), 평행이동(translation),
    # 회전(rotation), 상하좌우반전(flipping),
    # 이미지에 noise 추가(adding salt and pepper noise),
    # 명암조절(lighting condition),
    # object 중심으로 이미지 변환(or 확대)??(perspective transformation, object 중심으로 image를 crop)
    def __init__(self,
                 width,
                 height,
                 channel,
                 filename,
                 url,
                 dataRoot=None,
                 augmentedRate=3,
                 dir_to_save='images'):

        super().__init__(width,
                         height,
                         channel,
                         filename,
                         url,
                         dataRoot,
                         dir_to_save)

        self.augmentedRate = augmentedRate

    def readImages(self, path, label):

        return tf.image.decode_jpeg(
            tf.read_file(path),
            channels=self.channel,
            name='readRawImages'
        ), self.oneHotEncoding(label)

    def getDataset(self, batchsize=128):

        train, test, train_label, test_label = train_test_split(
            self.all_image_paths,
            self.all_image_labels,
            test_size=0.33,
            stratify=self.all_image_labels
        )

        testset = self.getRawImages(test, test_label)\
            .shuffle(len(test))\
            .repeat()\
            .batch(batchsize)\
            .prefetch(1)

        trainset = self.getRawImages(train,
                                     train_label)

        for i in range(self.augmentedRate):
            trainset = self.augmentedImageDataset(trainset,
                                                  train,
                                                  train_label)

        trainset = trainset \
            .shuffle(len(self.all_image_paths)*10*self.augmentedRate) \
            .repeat() \
            .batch(batchsize) \
            .prefetch(1)

        return trainset, testset

    def getRawImages(self, image_paths, image_labels):

        return tf.data.Dataset \
            .from_tensor_slices((image_paths,
                                 image_labels)) \
            .map(self.readImages, -1) \
            .map(self.getStandizeImage, -1) \
            .map(self.getResizedImage, -1)

    def getResizedImage(self, image, label):

        return tf.image \
                   .resize_images(image, [self.width,
                                          self.height]), label

    def getStandizeImage(self, image, label):

        return tf.image\
                   .per_image_standardization(image), label

    def augmentedImageDataset(self,
                              dataset,
                              train_image,
                              train_label):

        for augment in [self.getCentralImages,
                        self.getLeftRightFlipImages,
                        self.getUpDownFlipImages,
                        self.getRandomBrightImages,
                        self.getRandomContrastImages,
                        self.getRandomGammaImages,
                        # self.getRandomHueImages,
                        # self.getRandomSaturationImages,
                        self.getRotateImages]:

            dataset = dataset.concatenate(augment(train_image,
                                                  train_label))
        if self.channel == 1: return dataset

        for augment in [self.getRandomSaturationImages,
                        self.getRandomHueImages]:
            dataset = dataset.concatenate(augment(train_image,
                                                  train_label))
        return dataset

    def getCentralImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getCentralImage, -1)\
            .map(self.getResizedImage, -1)

    def _getCentralImage(self, image, label):

        return tf.image \
                   .central_crop(image,
                                 random.random()), label

    def getLeftRightFlipImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getLeftRightFlipImage, -1) \
            .map(self._getRotateImage, -1) \
            .map(self.getResizedImage, -1)

    def _getLeftRightFlipImage(self, image, label):

        return tf.image\
            .flip_left_right(image), label

    def getUpDownFlipImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getUpDownFlipImage, -1) \
            .map(self._getRotateImage, -1) \
            .map(self.getResizedImage, -1)

    def _getUpDownFlipImage(self, image, label):

        return tf.image \
                   .flip_up_down(image), label

    def getRotateImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getRotateImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRotateImage(self, image, label):

        return tf.image \
                   .rot90(image,
                          random.randrange(1, 91)), label

    def getRandomHueImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getRandomHueImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomHueImage(self, image, label):

        rand = random.random()
        if rand > 0.5: rand -= 0.5
        return tf.image \
                   .random_hue(image,
                               rand), label

    def getRandomSaturationImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getRandomSaturationImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomSaturationImage(self, image, label):

        return tf.image\
                   .adjust_saturation(image, random.random()), label

    def getRandomBrightImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getRandomBrightImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomBrightImage(self, image, label):

        return tf.image \
                   .adjust_brightness(image,
                                      random.random()), label

    def getRandomContrastImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self._getRandomContrastImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomContrastImage(self, image, label):

        return tf.image \
                   .adjust_contrast(image,
                                    random.random()), label

    def getRandomGammaImages(self, image, label):

        return tf.data.Dataset \
            .from_tensor_slices((image, label)) \
            .map(self.readImages, -1) \
            .map(self.getResizedImage, -1) \
            .map(self._getRandomGammaImage, -1)

    def _getRandomGammaImage(self, image, label):

        return tf.image \
                   .adjust_gamma(image,
                                 random.random()+0.5), label

    def getRandomCropImage(self):
        """
        이미지 크기를 구할 수 없어서 일단 보류
        :return:
        """

        _dataset = tf.data.Dataset\
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels))\
            .map(self.readImages, -1)\
            .map(self._getRandomCropImage, -1)

    def _getRandomCropImage(self, image, label, img_shape):

        w, h = img_shape
        cropSize = min(h, w, self.height, self.height)
        return tf.image \
                   .random_crop(image,
                                [cropSize, cropSize, 3]), label


if __name__ == '__main__':
    # tf.enable_eager_execution()
    # /101_ObjectCategories : 9145
    # test = ImageAugmentation(224, 224, 3,
    #                          '/101_ObjectCategories',
    #                          'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz',
    #                          augmentedRate=-1)
    # dt = test.getDataset(batchsize=128)
    # mnist = ImagePipeline(28, 28, 1,
    #                       '/mnistjpg',
    #                       None,
    #                       dataRoot='C:/Users/yun/Desktop/GoogleNet/images/mnistasjpg')

    path = 'C:/Users/yun/Desktop/GoogleNet/images/mnist'
    # images, labels = byte2np('C:/Users/yun/Desktop/GoogleNet/images/train-images.idx3-ubyte',
    #                          'C:/Users/yun/Desktop/GoogleNet/images/train-labels.idx1-ubyte')
    # np2image(images, labels, path)
    # images, labels = byte2np('C:/Users/yun/Desktop/GoogleNet/images/t10k-images.idx3-ubyte',
    #                          'C:/Users/yun/Desktop/GoogleNet/images/t10k-labels.idx1-ubyte')
    # np2image(images, labels, path)
    # getfile('/train_image', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist')
    # test = Byte2image('C:/Users/yun/Desktop/GoogleNet/images/train-images.idx3-ubyte',
    #                   'C:/Users/yun/Desktop/GoogleNet/images/train-labels.idx1-ubyte',
    #                   1, path)
    # test.transform()
    # test = Byte2image('C:/Users/yun/Desktop/GoogleNet/images/t10k-images.idx3-ubyte',
    #                   'C:/Users/yun/Desktop/GoogleNet/images/t10k-labels.idx1-ubyte',
    #                   1, path)
    # test.transform()
    test = ImageAugmentation(28, 28, 1, None, None, path, 1)
    test.getDataset(100)
    pass