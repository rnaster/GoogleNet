# TODO: image processing은 tensorflow으로...?
import os, json, collections

import tensorflow as tf


class ParsingJson:

    def __init__(self, filename):
        """

        :param filename:
        """

        self.filename = filename

        self.architecture = self.parsing()

    def parsing(self):
        if not self.isEmpty():
            raise ValueError('Not Found File')

        with open(self.filename) as f:
            architecture = json.load(f,
                                     object_pairs_hook=collections.OrderedDict)

        return architecture

    def isEmpty(self):
        """
        check existence of file

        :return:
        """

        return os.path.exists(self.filename)

    def specification(self):
        # TODO: count the number of parameter, print architecture
        pass


class ImageProcessing:
    """
    Image processing with tensorflow
    """
    def __init__(self, address):
        self.dataAddress = address

    def dataFlow(self):

        yield


if __name__ == '__main__':
    # tf.data.Dataset
    pass