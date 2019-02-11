import os
import json
import collections
import pathlib
import random
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

    data_root = tf.keras \
        .utils.get_file(path_to_download + filename,
                        url,
                        untar=True,
                        cache_dir=path_to_download,
                        cache_subdir=path_to_download)

    print('Success to download!')

    return pathlib.Path(data_root)


class ImagePipeline:
    """
    Image processing with tensorflow
    """
    # TODO: tfrecord 읽어 오기 혹은 raw image 사용
    def __init__(self,
                 filename,
                 url,
                 isRawImage=True,
                 dataRoot=None,
                 dir_to_save='images'):

        self.label_to_index = None
        self.all_image_labels = None
        self.labels = None
        self.height = None
        self.width = None
        self.channel = None
        self.dataRoot = getfile(filename, url, dir_to_save)\
            if dataRoot is None else dataRoot
        self.isRawImage = isRawImage

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

    # def getAllImageShape(self):
    #
    #     self.all_image_shape = [PIL.Image.open(path).size
    #                             for path in self.all_image_paths]

    def _labeling2images(self):

        label_names = sorted(item.name for item in self.dataRoot.glob('*/') if item.is_dir())
        self.label_to_index = dict((name, index) for index, name in enumerate(label_names))
        self.labels = len(label_names)
        self.all_image_labels = [self.label_to_index[pathlib.Path(path).parent.name]
                                 for path in self.all_image_paths]

    def readImages(self, path, label):

        image = tf.image.decode_jpeg(
            tf.read_file(path)
        )
        if self.isRawImage:
            image = tf.image.resize_images(image, [self.height, self.width])

        return self.imageStardization(image),\
               self.oneHotEncoding(label)

    def oneHotEncoding(self, label):

        return tf.one_hot(label, self.labels)

    def imageStardization(self, image):

        if self.isRawImage:

            return tf.image\
                .per_image_standardization(image)

        return image

    def getDataset(self,
                   height,
                   width,
                   channel=3,
                   batchsize=128):
        """

        :param height: image height to resize
        :param width: image width th resize
        :param channel:
        :param batchsize: batch size
        :return:
        """

        self._getAllImagePaths()
        self._labeling2images()

        self.height = height
        self.width = width
        self.channel = channel

        dataset = tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels))\
            .map(self.readImages, -1) \
            .shuffle(len(self.all_image_paths)) \
            .repeat() \
            .batch(batchsize) \
            .prefetch(1)

        return dataset


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
                 filename,
                 url,
                 dataRoot=None,
                 augmentedRate=3,
                 dir_to_save='images'):

        super().__init__(filename, url, False, dataRoot, dir_to_save)
        self.augmentedRate = augmentedRate

    def readImages(self, path, label):

        return tf.image.decode_jpeg(
            tf.read_file(path),
            channels=self.channel,
            name='readRawImages'
        ), self.oneHotEncoding(label)

    def getDataset(self, **kargs):

        self.height = kargs['height']
        self.width = kargs['width']
        self.channel = kargs['channel']
        batchSize = kargs['batchsize']

        dataset = self.getRawImages()

        for i in range(self.augmentedRate):
            print('%s-th augmentation' %i)
            # dataset = dataset.concatenate(
            #     self.augmentedImageDataset(dataset)
            # )
            dataset = self.augmentedImageDataset(dataset)

        return dataset \
            .shuffle(len(self.all_image_paths)) \
            .repeat() \
            .batch(batchSize) \
            .prefetch(1)

    def getRawImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
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

    def augmentedImageDataset(self, dataset):

        for augment in [self.getCentralImages,
                        # self.getNoiseImages,
                        self.getLeftRightFlipImages,
                        self.getUpDownFlipImages,
                        self.getRandomBrightImages,
                        self.getRandomContrastImages,
                        self.getRandomGammaImages,
                        self.getRandomHueImages,
                        self.getRandomSaturationImages,
                        self.getRotateImages]:

            dataset = dataset.concatenate(augment())

        return dataset

    def getCentralImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getCentralImage, -1)\
            .map(self.getResizedImage, -1)

    def _getCentralImage(self, image, label):

        return tf.image \
                   .central_crop(image,
                                 random.random()), label

    def getNoiseImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getNoiseImage, -1) \
            .map(self.getResizedImage, -1)

    def _getNoiseImage(self, image, label):

        return tf.image \
                   .adjust_jpeg_quality(image,
                                        random.randrange(0, 101)), label

    def getLeftRightFlipImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getLeftRightFlipImage, -1) \
            .map(self._getRotateImage, -1) \
            .map(self.getResizedImage, -1)

    def _getLeftRightFlipImage(self, image, label):

        return tf.image\
            .flip_left_right(image), label

    def getUpDownFlipImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getUpDownFlipImage, -1) \
            .map(self._getRotateImage, -1) \
            .map(self.getResizedImage, -1)

    def _getUpDownFlipImage(self, image, label):

        return tf.image \
                   .flip_up_down(image), label

    def getRotateImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getRotateImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRotateImage(self, image, label):

        return tf.image \
                   .rot90(image,
                          random.randrange(1, 91)), label

    def getRandomHueImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getRandomHueImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomHueImage(self, image, label):

        rand = random.random()
        if rand > 0.5: rand -= 0.5
        return tf.image \
                   .random_hue(image,
                               rand), label

    def getRandomSaturationImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getRandomSaturationImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomSaturationImage(self, image, label):

        return tf.image\
                   .adjust_saturation(image, random.random()), label

    def getRandomBrightImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getRandomBrightImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomBrightImage(self, image, label):

        return tf.image \
                   .adjust_brightness(image,
                                      random.random()), label

    def getRandomContrastImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1) \
            .map(self._getRandomContrastImage, -1) \
            .map(self.getResizedImage, -1)

    def _getRandomContrastImage(self, image, label):

        return tf.image \
                   .adjust_contrast(image,
                                    random.random()), label

    def getRandomGammaImages(self):

        return tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
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
    test = ImageAugmentation('/101_ObjectCategories',
                             'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz',
                             augmentedRate=2)
    dt = test.getDataset(height=224, width=224, channel=3, batchsize=128)

    pass