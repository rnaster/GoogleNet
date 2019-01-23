# TODO: 한글주석 -> 영어로, licence 만들기(MIT??)
import os
import json
import collections
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


class GoogleNet(ParsingJson):

    def __init__(self, filename, seed=None):
        super().__init__(filename)

        self.seed = seed

    def dataFlow(self):
        pass

    def train(self):
        # TODO: 학습 후 모델 저장
        # TODO: epoch마다 logging
        pass

    def predict(self):
        # 저장된 모델 불러와 test 데이터로 예측
        pass

    def buildGraph(self):
        # TODO: 불러온 architecture 기반으로 graph 빌드
        _image = self.architecture['image']

        width, height, channel = _image['width'], \
                                 _image['height'], \
                                 _image['channel']

        _label = self.architecture['label']

        labels, isEncoding = _label['class'], \
                             _label['isEncoding']

        # label이 one-hot encoding 되있으면 class 개수, 아니면 1
        y_dim = labels if isEncoding else 1

        layers = self.architecture['layers']

        inputX = tf.placeholder(tf.float32, [None, width, height, channel], name='inputX')
        outputY = tf.placeholder(tf.float32, [None, y_dim], name='outputY')

        # TODO: 필요한 인자가 있는지 확인 ex) conv: height, width, channel, stride
        for layer in layers:

            pass

    def getFilters(self, shape, name, is_uniform):
        """
        Initialize weight of filter
        :param shape:
        :param is_uniform:
        :param name:
        :return:
        """

        seed = self.seed

        initializer = tf.contrib.layers.xavier_initializer(is_uniform, seed)

        return tf.get_variable(initializer=initializer,
                               shape=shape,
                               name=name)

    def getWeights(self, shape, name):
        """
        Initialize normal distributed weight of fully connected layer,
        convolution bias
        :return:
        """

        seed = self.seed

        initializer = tf.initializers.random_normal(seed=seed)

        return tf.get_variable(initializer=initializer,
                               shape=shape,
                               name=name)

    def convolutionLayer(self,
                         input,
                         filter_height,
                         filter_width,
                         in_channels,
                         out_channels,
                         stride,
                         name,
                         padding='SAME',
                         is_uniform=False):
        """

        :param is_uniform: True : uniform / False : normal dist.
        :param input: input tensor
        :param filter_height: filter height
        :param filter_width: filter width
        :param in_channels: input tensor의 channel 갯수
        :param out_channels: output tensor의 channel 갯수
        :param stride: stride 설정
        :param padding: zero padding 여부
        :param name: layer 이름
        :return:
        """

        filter_name = 'weights'

        filter_shape = [
            filter_height,
            filter_width,
            in_channels,
            out_channels
        ]

        bias_name = 'bias'

        output_name = 'dropout'

        layer_name = name + '_layer'

        with tf.variable_scope(layer_name):
            filter = self.getFilters(filter_shape,
                                     filter_name,
                                     is_uniform)

            convolved = tf.nn.conv2d(input=input,
                                     filter=filter,
                                     strides=[1, stride, stride, 1],
                                     padding=padding,
                                     name='convolution')

            convolved_bias = self.getWeights([1, 1, 1, out_channels],
                                             bias_name)

            output = tf.nn.relu(convolved + convolved_bias,
                                'relu')

            output = self.dropOut(output, 0.3, output_name)  # TODO: 0.3 -> keep_prob으로 바꾸기

        return output

    @staticmethod
    def maxPooling(input,
                   filter_height,
                   filter_width,
                   stride,
                   name,
                   padding='SAME'):
        """

        :param input:
        :param filter_height:
        :param filter_width:
        :param stride:
        :param name:
        :param padding:
        :return:
        """

        return tf.nn.max_pool(input,
                              [1, filter_height, filter_width, 1],
                              [1, stride, stride, 1],
                              padding,
                              name=name)

    @staticmethod
    def batchNorm(input, name, is_train=True):
        return tf.layers.batch_normalization(input,
                                             training=is_train,
                                             name=name)

    def concatenation(self, output_list, name):
        """
        output format : [batch_size, width, height, channel]
        concatenation by axis=3

        :param output_list: tensor list
        :param name:
        :return:
        """

        return tf.concat(output_list, axis=3, name=name)

    @staticmethod
    def averagePooling(input,
                       filter_height,
                       filter_width,
                       stride,
                       name,
                       padding='SAME'):
        return tf.nn.avg_pool(input,
                              [1, filter_height, filter_width, 1],
                              [1, stride, stride, 1],
                              padding,
                              name=name)

    @staticmethod
    def fullyConnectedLayer(input, num_outputs, scope, activation_fn=tf.nn.relu):
        return tf.contrib.layers.fully_connected(input,
                                                 num_outputs,
                                                 activation_fn,
                                                 scope=scope)

    def getCrossEntropy(self, labels, outputs, name, isEncoding=False):
        """

        :param labels: actual Y
        :param outputs: 예측 값; WX + B
        :param name: ex) auxiliary_0 / main
        :param isEncoding: one-hot encoding 여부
        :return:
        """

        with tf.variable_scope(name):
            if isEncoding:
                return tf.nn.softmax_cross_entropy_with_logits_v2(labels,
                                                                  outputs,
                                                                  name='CrossEntropy')

            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels,
                                                                  outputs,
                                                                  'CrossEntropy')

    def getLoss(self, crossEntropy, name):
        with tf.variable_scope(name):
            return tf.math.reduce_mean(
                tf.math.reduce_sum(crossEntropy, 1, name='sum'),
                name='mean'
            )

    def dropOut(self, input, keep_prob, name):
        """

        :param input:
        :param keep_prob: element를 유지할 확률; 0.9 : 90% 확률로 element가 생존.
        :param name:
        :return:
        """

        return tf.nn.dropout(input,
                             keep_prob,
                             name=name)


if __name__ == '__main__':
    p = ParsingJson('C:/Users/yun/Desktop/GoogleNet/model/inception_v1/inception_v1.json')
    p.parsing()
    _g = p.architecture['graph']
    # for x in _g:
    #     print(_g[x])
    pass
