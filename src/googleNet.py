# TODO: 한글주석 -> 영어로, licence 만들기(MIT??)
import tensorflow as tf

from src.utils import ParsingJson, ImageProcessing


class GoogleNet:

    def __init__(self, filename, seed=None):

        self.seed = seed
        self.inputX = None
        self.outputY = None
        self.loss = None
        self._dropOut_rate_main = None
        self._dropOut_rate_auxiliary = None
        self.isTrain = None
        self.architecture = ParsingJson(filename).parsing()

        _image = self.architecture['image']

        self.width, self.height, self.channel = _image['width'], \
                                                _image['height'], \
                                                _image['channel']

    def train(self, filename, url):
        # TODO: 학습 후 모델 저장
        # TODO: epoch마다 logging

        _hyperparameter = self.architecture['hyperparameter']

        _optimizer = _hyperparameter['optimizer']

        _batchSize = _hyperparameter['batchSize']

        _epoch = _hyperparameter['epoch']

        _learning_rate = _hyperparameter['learningrate']

        _dropOut = _hyperparameter['dropout']
        _dropOut_rate_main = _dropOut['main']
        _dropOut_rate_auxiliary = _dropOut['auxiliary']

        if _optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(_learning_rate)\
                .minimize(self.loss)
        else:
            optimizer = tf.train.AdamOptimizer(_learning_rate)\
                .minimize(self.loss)

        dataset = ImageProcessing(filename, url, self.channel)\
            .getDataset(self.height, self.width, _batchSize)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _loss = None
            for epoch in range(1, _epoch+1):
                for batchX, batchY in dataset:
                    _, _loss = sess.run([optimizer, self.loss],
                                        feed_dict={self.inputX: batchX,
                                                   self.outputY: batchY,
                                                   self._dropOut_rate_main: _dropOut_rate_main,
                                                   self._dropOut_rate_auxiliary: _dropOut_rate_auxiliary,
                                                   self.isTrain: True})

                print('epoch: %s, train loss: %s' % (epoch, _loss))

    def predict(self):
        # 저장된 모델 불러와 test 데이터로 예측
        pass

    def buildGraph(self):

        _label = self.architecture['label']

        num_labels = _label['class']

        self.inputX = tf.placeholder(tf.float32,
                                     [None, self.width, self.height, self.channel],
                                     name='inputX')
        self.outputY = tf.placeholder(tf.float32,
                                      [None, num_labels],
                                      name='outputY')

        self._dropOut_rate_main = tf.placeholder(tf.float32, name='dropOutMain')
        self._dropOut_rate_auxiliary = tf.placeholder(tf.float32, name='dropOutAuxiliary')

        self.isTrain = tf.placeholder(tf.bool, name='isTrain')

        layers = self.architecture['layers']

        # TODO: 필요한 인자가 있는지 확인 ex) conv: height, width, channel, stride
        previous_layer = self.inputX

        loss_list = []
        for layer in layers:
            _layer = layers[layer]
            with tf.variable_scope(layer):

                for inner_layer in _layer:
                    _inner_layer = _layer[inner_layer]
                    with tf.variable_scope(inner_layer):

                        if 'convolution' in inner_layer:
                            previous_layer = self.convolution(previous_layer,
                                                              _inner_layer)
                        elif 'maxpooling' in inner_layer:
                            _height, _width, _stride, _padding = _inner_layer['height'], \
                                                                 _inner_layer['width'], \
                                                                 _inner_layer['stride'], \
                                                                 _inner_layer['padding']

                            previous_layer = self.maxPooling(previous_layer,
                                                             _height,
                                                             _width,
                                                             _stride,
                                                             'maxPooling',
                                                             _padding)
                        elif 'batchNorm' in inner_layer:
                            if _inner_layer:
                                previous_layer = self.batchNorm(previous_layer)

                        elif 'inception' in inner_layer:
                            previous_layer = self.inception(previous_layer, _inner_layer)

                        elif 'getLoss' in inner_layer:
                            loss_list.append(self.getLoss(previous_layer, self.outputY, _inner_layer))

        self.loss = self.getTotalLoss(loss_list)

    def getTotalLoss(self, loss_list):

        return tf.math.add_n(loss_list, name='total_loss')

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

    def convolution(self, input, convs):
        _previous_layer = input

        for conv in convs:
            _conv = convs[conv]
            _height, _width, _out_channel, _stride, _padding = _conv['height'], \
                                                               _conv['width'], \
                                                               _conv['channel'], \
                                                               _conv['stride'], \
                                                               _conv['padding']
            _previous_layer = self.conv(_previous_layer,
                                        _height,
                                        _width,
                                        _out_channel,
                                        _stride,
                                        conv,
                                        _padding)
            _in_channel = _out_channel

        return _previous_layer

    def conv(self,
             input,
             filter_height,
             filter_width,
             out_channels,
             stride,
             name,
             padding='SAME',
             isAuxiliary=False,
             is_uniform=False):
        """

        :param isAuxiliary:
        :param is_uniform: True : uniform / False : normal dist.
        :param input: input tensor
        :param filter_height: filter height
        :param filter_width: filter width
        :param out_channels: output tensor의 channel 갯수
        :param stride: stride 설정
        :param padding: zero padding 여부
        :param name: layer 이름
        :return:
        """

        _, _, _, in_channels = input.shape

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

            if isAuxiliary:
                output = self.dropOut(output, self._dropOut_rate_auxiliary, output_name)
            else:
                output = self.dropOut(output, self._dropOut_rate_main, output_name)

        return output

    def inception(self, input, config):
        """

        :param input:
        :param config: configuration for inception module
        :return:
        """

        output_list = []
        _previous_layer = input
        for convs in config:
            _layer = config[convs]

            with tf.variable_scope(convs):
                for conv in _layer:
                    _inner_layer = _layer[conv]

                    with tf.variable_scope(conv):
                        if 'conv' in conv:
                            _height, _width, _stride, _padding, _out_channel = self._getLayerParams('conv',
                                                                                                    _inner_layer)

                            _previous_layer = self.conv(_previous_layer,
                                                        _height,
                                                        _width,
                                                        _out_channel,
                                                        _stride,
                                                        conv,
                                                        _padding)
                        elif 'maxpooling' in conv:
                            _height, _width, _stride, _padding, _ = self._getLayerParams('maxpooling',
                                                                                         _inner_layer)

                            _previous_layer = self.maxPooling(_previous_layer,
                                                              _height,
                                                              _width,
                                                              _stride,
                                                              'maxPooling',
                                                              _padding)

            output_list.append(_previous_layer)

        return self.concatenation(output_list)

    def _getLayerParams(self, layerType, layerInfo):
        if layerType == 'conv':

            return layerInfo['height'], \
                   layerInfo['width'], \
                   layerInfo['stride'], \
                   layerInfo['padding'], \
                   layerInfo['channel']

        elif layerType == 'fc':

            return layerInfo['num_outputs'], \
                   layerInfo['activation_fn'], \
                   None, \
                   None, \
                   None

        else:

            return layerInfo['height'], \
                   layerInfo['width'], \
                   layerInfo['stride'], \
                   layerInfo['padding'], \
                   None

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

    def batchNorm(self, input):

        return tf.layers.batch_normalization(input,
                                             training=self.isTrain)

    def concatenation(self, output_list):
        """
        output format : [batch_size, width, height, channel]
        concatenation by axis=3
        with tf.variable_scope(name=name) 내부에서 호출해야함

        :param output_list: tensor list
        :return:
        """

        return tf.concat(output_list, axis=3, name='concat')

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
    def fullyConnectedLayer(input,
                            num_outputs,
                            scope,
                            activation_fn='relu'):
        """

        :param input:
        :param num_outputs:
        :param scope:
        :param activation_fn:
        :return:
        """

        if activation_fn == 'relu':
            _activation_fn = tf.nn.relu
        else:
            _activation_fn = None

        return tf.contrib.layers.fully_connected(input,
                                                 num_outputs,
                                                 _activation_fn,
                                                 scope=scope)

    def getLoss(self, input, output, config):
        """

        :param input:
        :param output:
        :return:
        """

        if config['auxiliary']:
            name = 'auxiliary'
            rate = 0.3
        else:
            name = 'main'
            rate = 1

        _loss_layer = config['loss']

        _previous_layer = input

        with tf.variable_scope(name):
            for _layer in _loss_layer:
                _inner_layer = _loss_layer[_layer]

                if 'averagepooling' in _layer:
                    _height, _width, _stride, _padding, _ = self._getLayerParams('averagepooling',
                                                                                 _inner_layer)

                    _previous_layer = self.averagePooling(_previous_layer,
                                                          _height,
                                                          _width,
                                                          _stride,
                                                          'averagePooling',
                                                          _padding)

                elif 'conv' in _layer:
                    _height, _width, _stride, _padding, _out_channel = self._getLayerParams('conv',
                                                                                            _inner_layer)

                    _previous_layer = self.conv(_previous_layer,
                                                _height,
                                                _width,
                                                _out_channel,
                                                _stride,
                                                _layer,
                                                _padding,
                                                isAuxiliary=True)

                elif 'fc' in _layer:
                    _num_outputs, _activation_fn, _, _, _ = self._getLayerParams('fc',
                                                                                 _inner_layer)

                    _previous_layer = self.fullyConnectedLayer(_previous_layer,
                                                               _num_outputs,
                                                               _layer,
                                                               _activation_fn)
                elif 'flatten' in _layer:
                    _previous_layer = self.flatten(_previous_layer)

            _previous_layer = self.softmax(_previous_layer)

            return self.getCrossEntropy(_previous_layer,
                                        output) * rate

    def flatten(self, input):

        _, w, h, c = input.shape

        return tf.reshape(input, [-1, w * h * c], name='flatten')

    def softmax(self, input):

        return tf.nn.softmax(input, name='softmax')

    def getCrossEntropy(self, labels, outputs):
        """

        :param labels: actual Y
        :param outputs: 예측 값(softmax); WX + B
        :param name: ex) auxiliary_0 / main
        :return:
        """

        return -tf.reduce_mean(
            tf.reduce_sum(tf.log(outputs) * labels,
                          axis=1),
            name='crossEntropy'
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
    # test = GoogleNet('C:/Users/yun/Desktop/GoogleNet/model/inception_v1/inception_v1.json')
    # test.buildGraph()
    # dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    # dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    # dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    # print(dataset3.output_shapes)
    # print(dataset3.output_types)
    pass