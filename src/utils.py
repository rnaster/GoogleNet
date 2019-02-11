import os
import json
import collections
import pathlib
import random
import tempfile
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

        self.getAllImagePaths()
        self.labeling2images()

    def getAllImagePaths(self):

        all_image_paths = list(self.dataRoot.glob('*/*'))
        self.all_image_paths = [str(path)
                                for path in all_image_paths
                                if 'Thumbs.db' not in str(path)]

    # def getAllImageShape(self):
    #
    #     self.all_image_shape = [PIL.Image.open(path).size
    #                             for path in self.all_image_paths]

    def labeling2images(self):

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

    def getDataset(self, height, width, channel=3, batchSize=128):
        """

        :param height: image height to resize
        :param width: image width th resize
        :param channel:
        :param batchSize: batch size
        :return:
        """

        self.getAllImagePaths()
        self.labeling2images()

        self.height = height
        self.width = width
        self.channel = channel

        dataset = tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels))\
            .map(self.readImages, -1) \
            .shuffle(len(self.all_image_paths)) \
            .repeat() \
            .batch(batchSize) \
            .prefetch(1)

        return dataset


class ImageAugmentation(ImagePipeline):
    """
    augmentation, pre-processing of image
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

        self.parentDir = os.path.dirname(self.dataRoot)
        self.dirName = os.path.basename(self.dataRoot) + '_Augmented'
        self.newDirName = self.parentDir + self.dirName

        os.makedirs(self.newDirName, exist_ok=True)

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def readImages(self, path, label):

        return tf.image.decode_jpeg(
            tf.read_file(path),
            channels=self.channel,
            name='readRawImages'
        ), label

    def getDataset(self, **kargs):

        self.height = kargs['height']
        self.width = kargs['width']
        self.channel = kargs['channel']

        for _ in range(self.augmentedRate):
            self.augmentImages()

    def augmentImages(self):

        self.getResizedImage()

    def serialize(self, image, label):

        feature = {
            'image': self._bytes_feature(image),
            'label': self._int64_feature(label)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def getResizedImage(self):

        filename = tempfile.mktemp('.tfrecord',
                                   dir=self.newDirName)
        _dataset = tf.data.Dataset \
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels)) \
            .map(self.readImages, -1)\
            .map(self._getResizedImage, -1)\
            .map(self.serialize, -1)
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(_dataset)

    def _getResizedImage(self, image, label):

        return tf.image \
                   .resize_images(image, [self.width,
                                          self.height]), label

    def getRandomCropImage(self):
        """
        이미지 크기를 구할 수 없어서 일단 보류
        :return:
        """
        filename = tempfile.mktemp('.tfrecord',
                                   dir=self.newDirName)
        _dataset = tf.data.Dataset\
            .from_tensor_slices((self.all_image_paths,
                                 self.all_image_labels))\
            .map(self.readImages, -1)\
            .map(self._getRandomCropImage, -1)\
            .map(self.serialize, -1)

        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.writer(_dataset)

    def _getRandomCropImage(self, image, label, img_shape):

        w, h = img_shape
        cropSize = min(h, w, self.height, self.height)
        return tf.image \
                   .random_crop(image,
                                [cropSize, cropSize, 3]), label

    def getRandomNoise(self, image, label):

        rand = random.random()
        if rand > 0.5: return image, label
        return tf.image\
                   .adjust_jpeg_quality(image, random.randrange(0, 101)), label

    def getRandomFlip(self, image, label):

        rand = random.random()
        if rand > 2/3: return image, label
        if rand > 1/3: return tf.image\
                                  .random_flip_left_right(image), label
        return tf.image.random_flip_up_down(image), label

    def getRotation(self, image, label):

        rand = random.random()
        if rand > 0.5: return image, label
        return tf.image.rot90(image, random.randrange(1, 91)), label

    def getRandomHue(self, image, label):

        rand = random.random()
        if rand > 0.5: return image, label
        return tf.image.random_hue(image, rand), label

    def getRandomSaturation(self, image, label):

        rand = random.random()
        if rand > 0.5: return image, label
        return tf.image\
                   .adjust_saturation(image, random.random()), label

    def getRandomBrightness(self, image, label):

        rand = random.random()
        if rand > 0.5: return image, label
        return tf.image\
                   .adjust_brightness(image, random.random()), label

    def getRandomContrast(self, image, label):

        rand = random.random()
        if rand > 0.5: return image, label
        return

    def getRandomGamma(self, image, label):
        return


if __name__ == '__main__':
    # tf.enable_eager_execution()
    # /101_ObjectCategories : 9145
    # test = ImageAugmentation('/101_ObjectCategories',
    #                          'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz')
    # test.getDataset(height=224, width=224, channel=3)

    pass