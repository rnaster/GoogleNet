# TODO: 한글주석 -> 영어로, licence 만들기(MIT??)
# TODO: predict 구현
import os
import math
import tensorflow as tf

from src.utils import ParsingJson, ImagePipeline, ImageAugmentation


class GoogleNet:

    def __init__(self,
                 model_json,
                 filename=None,
                 url=None,
                 dataRoot=None,
                 augmentedRate=3,
                 seed=None):

        self.model_address = os.path.dirname(model_json)
        self.architecture = ParsingJson(model_json).parsing()
        self.filename = filename
        self.url = url
        self.dataRoot = dataRoot
        self.augmentedRate = augmentedRate
        self.seed = seed

        self.trainset = None
        self.testset = None
        self.handle = None
        self.loss = None
        self.labels = None
        self.correct = None
        self._batchsize = None
        self._dropOut_rate_main = None
        self._dropOut_rate_aux = None
        self.isTrain = None

        _image = self.architecture['image']
        self.width, self.height, self.channel = _image['width'], \
                                                _image['height'], \
                                                _image['channel']

        _hyperparameter = self.architecture['hyperparameter']

        self.batchSize = _hyperparameter['batchSize']
        self.learning_rate = _hyperparameter['learningrate']

        self.all_image_paths = None
        self.all_image_labels = None

    def getCorrect(self, predicted, actual):
        """

        :param predicted: softmax value
        :param actual: one hot encoding value
        :return: correct 1-D vector
        """

        predictedLabel = tf.argmax(predicted,
                                   axis=1,
                                   name='predictedLabel')

        return tf.math.reduce_sum(
            tf.one_hot(predictedLabel, self.labels) * actual,
            axis=1,
            name='correctVector'
        )

    def train(self):
        # TODO: 학습 후 모델 저장
        # TODO: epoch마다 logging

        _hyperparameter = self.architecture['hyperparameter']

        _optimizer = _hyperparameter['optimizer']

        _epoch = _hyperparameter['epoch']

        _dropOut = _hyperparameter['dropout']
        _dropOut_rate_main = _dropOut['main']
        _dropOut_rate_aux = _dropOut['auxiliary']

        learning_rate = tf.placeholder(tf.float32)

        if _optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate) \
                .minimize(self.loss)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate) \
                .minimize(self.loss)

        stepsize = self.getStepsize()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # tf.summary.FileWriter(self.model_address,
            #                       sess.graph);exit()
            _loss = None
            train_handle = sess.run(
                self.trainset
                    .make_one_shot_iterator()
                    .string_handle())
            for epoch in range(1, _epoch + 1):
                correct = 0
                _batchsize = 0
                for step in range(stepsize):
                    _, _loss, _correct = sess.run([optimizer,
                                                   self.loss,
                                                   self.correct],
                                                  feed_dict={
                                                      learning_rate: self.learning_rate,
                                                      self.isTrain: True,
                                                      self._dropOut_rate_main: _dropOut_rate_main,
                                                      self._dropOut_rate_aux: _dropOut_rate_aux,
                                                      self.handle: train_handle})
                    correct += sum(_correct)
                    _batchsize += len(_correct)

                print('epoch: %s,'
                      ' train case: %s,'
                      ' train accuracy: %s,'
                      ' train loss: %s' % (epoch,
                                           _batchsize,
                                           correct / _batchsize,
                                           _loss))

                if epoch % 4 == 0: self.learning_rate *= 0.96

            saver.save(sess,
                       save_path=self.model_address + '/model')

    def getStepsize(self, isTrain=True):

        rate = 2 if isTrain else 1
        totalDataSize = len(self.all_image_paths)
        if self.augmentedRate > 0:
            stepsize = math.ceil(
                totalDataSize // 3 * rate * \
                self.augmentedRate * 8 // \
                self.batchSize
            )
        else:
            stepsize = math.ceil(
                totalDataSize // 3 * rate // \
                self.batchSize
            )

        return stepsize

    def predict(self, address):
        # 저장된 모델 불러와 test 데이터로 예측
        # address: test dataset address

        stepsize = self.getStepsize(False)
        correct = 0
        _batchsize = 0

        with tf.Session() as sess:
            importer = tf.train.import_meta_graph(address+'/model.meta')
            importer.restore(sess, tf.train.latest_checkpoint(address))
            graph = tf.get_default_graph()
            correctVector = graph.get_tensor_by_name('layer3/getLoss/main/correctVector:0')
            test_handle = sess.run(
                self.testset
                    .make_one_shot_iterator()
                    .string_handle())
            for _ in range(stepsize):
                _correct = sess.run(correctVector,
                                    feed_dict={
                                        self.handle: test_handle,
                                        self.isTrain: False,
                                        self._dropOut_rate_main: 1,
                                        self._dropOut_rate_aux: 1})
                correct += sum(_correct)
                _batchsize += len(_correct)
            print('test case: %s, test accuracy: %s'
                  % (_batchsize,
                     correct / _batchsize))

    def dataFlow(self):

        if self.augmentedRate > 0:
            imageProcessing = ImageAugmentation(
                self.width,
                self.height,
                self.channel,
                self.filename,
                self.url,
                self.dataRoot,
                self.augmentedRate
            )
        else:
            imageProcessing = ImagePipeline(
                self.width,
                self.height,
                self.channel,
                self.filename,
                self.url,
                self.dataRoot
            )

        self.dataRoot = imageProcessing.dataRoot

        trainset, testset = imageProcessing \
            .getDataset(batchsize=self.batchSize)

        self.all_image_paths = imageProcessing.all_image_paths
        self.labels = imageProcessing.labels

        return trainset, testset

    def buildGraph(self):

        tf.reset_default_graph()

        trainset, testset = self.dataFlow()
        self.trainset = trainset
        self.testset = testset

        self.handle = tf.placeholder(tf.string, [], name='handle')
        iterator = tf.data.Iterator \
            .from_string_handle(self.handle,
                                trainset.output_types,
                                trainset.output_shapes)
        inputX, outputY = iterator.get_next()

        self._dropOut_rate_main = tf.placeholder(tf.float32,
                                                 name='dropOutMain')
        self._dropOut_rate_aux = tf.placeholder(tf.float32,
                                                name='dropOutAux')

        self.isTrain = tf.placeholder(tf.bool,
                                      name='isTrain')

        layers = self.architecture['layers']

        # TODO: 필요한 인자가 있는지 확인 ex) conv: height, width, channel, stride
        previous_layer = inputX

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

                            previous_layer = self.maxPooling(
                                previous_layer,
                                _height,
                                _width,
                                _stride,
                                'maxPooling',
                                _padding)

                        elif 'inception' in inner_layer:
                            previous_layer = self.inception(previous_layer,
                                                            _inner_layer)

                        elif 'batchNorm' in inner_layer:
                            previous_layer = self.batchNorm(previous_layer)

                        elif 'getLoss' in inner_layer:
                            loss_list.append(self.getLoss(previous_layer,
                                                          outputY,
                                                          _inner_layer))
                    # print(previous_layer)

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

        initializer = tf.contrib \
            .layers \
            .xavier_initializer(is_uniform,
                                seed)

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
        initializer = tf.initializers \
            .random_normal(seed=seed,
                           stddev=0.01)

        return tf.get_variable(initializer=initializer,
                               shape=shape,
                               name=name,
                               trainable=True)

    def convolution(self, input, convs):
        _previous_layer = input

        for conv in convs:
            _conv = convs[conv]
            _height, _width, _out_channel, _stride, _padding, _batchNorm = _conv['height'], \
                                                                           _conv['width'], \
                                                                           _conv['channel'], \
                                                                           _conv['stride'], \
                                                                           _conv['padding'], \
                                                                           _conv['batchNorm']
            _previous_layer = self.conv(
                _previous_layer,
                _height,
                _width,
                _out_channel,
                _stride,
                conv,
                _padding,
                _batchNorm)
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
             batchNorm=False,
             isAuxiliary=False,
             is_uniform=False):
        """

        :param isAuxiliary:
        :param batchNorm:
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

            convolved = tf.nn.conv2d(
                input=input,
                filter=filter,
                strides=[1, stride, stride, 1],
                padding=padding,
                name='convolution')

            convolved_bias = self.getWeights(
                [1, 1, 1, out_channels],
                bias_name)

            affine = convolved + convolved_bias

            if batchNorm:
                output = tf.nn.relu(
                    self.batchNorm(affine),
                    'relu')
            else:
                output = tf.nn.relu(affine,
                                    'relu')

            if isAuxiliary:
                output = self.dropOut(
                    output,
                    self._dropOut_rate_aux,
                    output_name)
            else:
                output = self.dropOut(
                    output,
                    self._dropOut_rate_main,
                    output_name)

        return output

    def inception(self, input, config):
        """

        :param input:
        :param config: configuration for inception module
        :return:
        """

        output_list = []
        for convs in config:
            _layer = config[convs]
            _previous_layer = input

            with tf.variable_scope(convs):
                for conv in _layer:
                    _inner_layer = _layer[conv]

                    with tf.variable_scope(conv):
                        if 'conv' in conv:
                            _height, _width, _stride, _padding, _out_channel = self._getLayerParams('conv',
                                                                                                    _inner_layer)

                            _previous_layer = self.conv(
                                _previous_layer,
                                _height,
                                _width,
                                _out_channel,
                                _stride,
                                conv,
                                _padding)

                        elif 'maxpooling' in conv:
                            _height, _width, _stride, _padding, _ = self._getLayerParams('maxpooling',
                                                                                         _inner_layer)

                            _previous_layer = self.maxPooling(
                                _previous_layer,
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

        return tf.nn.max_pool(
            input,
            [1, filter_height, filter_width, 1],
            [1, stride, stride, 1],
            padding,
            name=name)

    def batchNorm(self, input):

        return tf.layers \
            .batch_normalization(input,
                                 training=self.isTrain)

    def concatenation(self, output_list):
        """
        output format : [batch_size, width, height, channel]
        concatenation by axis=3
        with tf.variable_scope(name=name) 내부에서 호출해야함

        :param output_list: tensor list
        :return:
        """

        return tf.concat(output_list,
                         axis=3,
                         name='concat')

    @staticmethod
    def averagePooling(input,
                       filter_height,
                       filter_width,
                       stride,
                       name,
                       padding='SAME'):

        return tf.nn.avg_pool(
            input,
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

        return tf.contrib \
            .layers \
            .fully_connected(input,
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

                    _previous_layer = self.averagePooling(
                        _previous_layer,
                        _height,
                        _width,
                        _stride,
                        'averagePooling',
                        _padding)

                elif 'conv' in _layer:
                    _height, _width, _stride, _padding, _out_channel = self._getLayerParams('conv',
                                                                                            _inner_layer)

                    _previous_layer = self.conv(
                        _previous_layer,
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

                    _previous_layer = self.fullyConnectedLayer(
                        _previous_layer,
                        _num_outputs,
                        _layer,
                        _activation_fn)

                elif 'flatten' in _layer:
                    _previous_layer = self.flatten(_previous_layer)

            _previous_layer = self.softmax(_previous_layer)

            if not config['auxiliary']:
                self.correct = self.getCorrect(_previous_layer,
                                               output)

            return self.getCrossEntropy(_previous_layer,
                                        output) * rate

    def flatten(self, input):

        _, w, h, c = input.shape

        return tf.reshape(input,
                          [-1, w * h * c],
                          name='flatten')

    def softmax(self, input):

        return tf.nn.softmax(input,
                             name='predicted')

    def getCrossEntropy(self, labels, outputs):
        """

        :param labels: actual Y
        :param outputs: 예측 값(softmax); WX + B
        :return:
        """

        return -tf.reduce_mean(
            tf.reduce_sum(tf.log(outputs + 1e-7) * labels,
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
    # test = GoogleNet('C:/Users/yun/Desktop/GoogleNet/model/101_objectCategories/test.json',
    #                  '/101_ObjectCategories',
    #                  'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz',
    #                  augmentedRate=-1)
    # test.buildGraph()
    # test.train()
    # test.train()
    # test.train()
    # dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    # dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    # dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    # print(dataset3.output_shapes)
    # print(dataset3.output_types)
    # a = tf.placeholder(tf.float32)
    # _dataset = tf.data.Dataset.from_tensor_slices(a).make_initializable_iterator()
    # X = _dataset.get_next()
    # Y = X * 2 + 1
    # with tf.Session() as sess:
    #     sess.run(_dataset.initializer, feed_dict={a: [3, 4, 5]})
    #     print(sess.run(Y))
    #     print(sess.run(Y))
    #     sess.run(_dataset.initializer, feed_dict={a: [3, 4, 5]})
    #     print(sess.run(Y))
    #     print(sess.run(Y))
    test = GoogleNet('C:/Users/yun/Desktop/GoogleNet/model/mnist/v2/architecture.json',
                     dataRoot='C:/Users/yun/Desktop/GoogleNet/images/mnist',
                     augmentedRate=1)
    test.buildGraph()
    test.train()
    # test.predict('C:/Users/yun/Desktop/GoogleNet/model/mnist/v3')
    pass
