import os
import time

import numpy as np
from keras.layers import Embedding, LSTM
from keras.layers.core import Dense, Activation
from keras.metrics import fbeta_score
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def fbeta_custom_score(y_true, y_pred) :
    return fbeta_score(y_true, y_pred, beta = 0)


class WordDNNRecaser(object) :
    def __init__(self,borders=2,name=None):
        self.border = borders
        self.name = name
        self.model = self.__init_model()
        self.__load_model()

    def learn(self, elements, epochs = 4) :
        learn_text, learn_result = self.__format_text(elements)

        data = [learn_text, learn_result]

        self.model = self.__run_network(data, self.model, epochs = epochs)

        self.__save_model()

    def predict(self, elements) :

        test_text, test_result = self.__format_text(elements)

        y = self.model.predict(test_text)

        y_classes = np_utils.categorical_probas_to_classes(y)

        return y_classes

    # ===============
    # PRIVATE METHODS
    # ===============

    def __init_model(self) :
        start_time = time.time()

        print('Compiling Model ... ')
        model = Sequential()
        # set input shape
        model.add(Embedding(input_dim = 10000, output_dim = 1000, input_shape = (1 + self.border * 2, 1)))

        model.add(LSTM(500))

        model.add(Dense(200))
        # shape the output
        model.add(Dense(3))
        model.add(Activation('softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                      metrics = [fbeta_custom_score])
        print('Model compield in {0} seconds'.format(time.time() - start_time))
        return model

    def __run_network(self, data, model, epochs = 10) :
        try :
            start_time = time.time()

            X_train, y_train = data

            print('Training model...')
            class_weight = {0 : 1.,
                            1 : 1.,
                            2 : 1.
                            }

            model.fit(X_train, y_train, validation_split = 0.2, nb_epoch = epochs, batch_size = 256,
                      verbose = 1, shuffle = False, class_weight = class_weight)

            print("Training duration : {0}".format(time.time() - start_time))

            return model
        except KeyboardInterrupt :
            print(' KeyboardInterrupt')
            return model

    def __load_model(self):
        if self.name != None:
            base_path = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(base_path, "..",
                                                "..", "..", "..",
                                                "..", "resources",
                                                "dnn",self.name+".h5"))

            if os.path.isfile(path) :
                self.model.load_weights(path)

    def __save_model(self) :
        if self.name != None:
            base_path = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(base_path, "..",
                                                "..", "..", "..",
                                                "..", "resources",
                                                "models","dnn",
                                                self.name+".h5"))

            self.model.save_weights(path)

    def __format_text(self, elements) :
        source = []
        result = []
        for element in elements :
            source.append(element.id + 1)
            result.append(element.operation)

        len_source = len(source)
        source = np.array(source)
        source = np.reshape(source, (len_source, 1))

        source_data = source

        zeros = np.zeros((self.border, 1))

        source = np.append(zeros, source, 0)
        source = np.append(source, zeros, 0)

        for i in range(1, self.border + 1) :
            j = self.border + i
            k = self.border - i
            source_data = np.append(source[k :(len_source + k)], source_data, 1)
            source_data = np.append(source_data, source[j :(len_source + j)], 1)

        encoder = LabelEncoder()
        encoder.fit([0, 1, 2])
        encoded_Y = encoder.transform(result)
        # convert integers to dummy variables (i.e. one hot encoded)
        result = np_utils.to_categorical(encoded_Y, 3)

        return source_data, result
