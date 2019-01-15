# TODO: 한글주석 -> 영어로, licence
import tensorflow as tf


class GoogleNet:

    def __init__(self, address=None, seed=None):
        """

        :param address: json 형태의 architecture 가 있는 파일 주소
        """
        self.address = address
        self.seed = seed

    def config(self):
        # TODO: parameter 개수, architecture 보여주기
        pass

    def dataFlow(self):
        pass

    def buildGraph(self):
        # TODO: 불러온 architecture 기반으로 graph 구성
        pass

    def getFilter(self, shape, name, is_uniform):
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

    def getWeight(self, shape, name):
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

        filter_name = name + '_weights'

        filter_shape = [
            filter_height,
            filter_width,
            in_channels,
            out_channels
        ]

        bias_shape = [
            filter_height,
            filter_width
        ]

        bias_name = name + '_bias'

        output_name = name + '_output'

        layer_name = name + '_layer'

        with tf.variable_scope(layer_name):

            filter = self.getFilter(filter_shape,
                                    filter_name,
                                    is_uniform)

            convolved = tf.nn.conv2d(input=input,
                                     filter=filter,
                                     strides=[1, stride, stride, 1],
                                     padding=padding,
                                     name='convolution')

            convolved_bias = self.getWeight(bias_shape,
                                            bias_name)

            output = tf.nn.relu(convolved + convolved_bias,
                                output_name)

            return output

    def maxPooling(self):
        pass

    def batchNorm(self):
        pass

    def concatenation(self):
        pass

    def classifier(self):
        pass

    def averagePooling(self):
        pass

    def fullyConnectedLayer(self):
        pass

    def softMaxActivation(self):
        pass

    def dropOut(self):
        pass


if __name__ == '__main__':

    with tf.variable_scope('foo'):
        a = tf.get_variable('bar', [1])
    with tf.variable_scope('goo'):
        b = tf.get_variable('bar', [2])
    print(a, b)