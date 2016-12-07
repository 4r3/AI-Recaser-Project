import os
import time
import numpy as np
from keras.utils import np_utils
from keras.layers import Embedding, LSTM,SimpleRNN,GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation
from keras.metrics import fbeta_score
from keras.utils.data_utils import get_file
from src.fr.enssat.recaser.tests.ParserTest import getAbsolutePath
from src.fr.enssat.recaser.parser.Parser import Parser

def fbeta_custom_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0)


class CharDNNRecaser(object) :
    def __init__(self):
        self.border = 1;
        self.model = self.__init_model()

    def learn(self, ressources_path = "corpus_1"):
        path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
        text = open(path).read()
        elements = self.__get_elements_from_text(text)
        #elements = self.__get_elements_from_file(ressources_path + "/corpus")
        learn_text, learn_result = self.__format_text(elements)

        data = [learn_text, learn_result]

        self.model = self.__run_network(data, self.model, epochs=50)

    def predict(self, text):

        elements = self.__get_elements_from_text(text)

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
        #set input shape
        model.add(Embedding(input_dim = 1000, output_dim = 1000, input_shape = (1+self.border*2,1)))

        model.add(LSTM(500))

        model.add(Dense(50))
        # shape the output
        model.add(Dense(2))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=[fbeta_custom_score])
        print('Model compield in {0} seconds'.format(time.time() - start_time))
        return model

    def __run_network(self, data, model, epochs = 20) :
        try :
            start_time = time.time()

            X_train, y_train = data

            print('Training model...')
            class_weight = {0: 1.,
                            1: 1}

            model.fit(X_train, y_train,validation_split=0.2, nb_epoch = epochs, batch_size = 256,
                      verbose = 2, shuffle = False, class_weight=class_weight)

            print("Training duration : {0}".format(time.time() - start_time))

            return model
        except KeyboardInterrupt :
            print(' KeyboardInterrupt')
            return model

    def __load_model(self) :
        model_path = "model.json"

        if os.path.isfile(model_path) :
            with open(model_path, 'r') as model_file :
                model_json = model_file.read()
            model = model_from_json(model_json)
            model.compile(loss = 'mape', optimizer = 'rmsprop',
                          metrics = [fbeta_custom_score])
        else:
            model = self.__init_model()
            with open(model_path, 'w') as model_file :
                model_file.write(model.to_json())

        if os.path.isfile('weight.h5') :
            model.load_weights('weight.h5', by_name = True)

        return model

    def __save_model(self):
        model_path = "model.json"
        with open(model_path, 'w') as model_file:
            model_file.write(self.model.to_json())
        if os.path.isfile('weight.h5') :
            self.model.load_weights('weight.h5', by_name = True)

    def __format_text(self, elements):
        source = []
        result = []
        for element in elements:
            source.append(element.value)
            result.append(element.operation)

        for ndx, member in enumerate(source) :
            source[ndx] = ord(source[ndx])%1000

        source = np.array(source)
        source = np.reshape(source, (len(source), 1))

        source_data = source
        zeros = np.zeros((self.border,1))
        source = np.append(zeros,source,0)
        source = np.append(zeros,source,0)
        #source = np.append(source,zeros,0)

        source_data = np.append(source[1:-1],source_data,1)
        source_data = np.append(source[0:-2],source_data,1)
        #source_data = np.append(source_data,source[2:],1)

        encoder = LabelEncoder()
        encoder.fit(result)
        encoded_Y = encoder.transform(result)
        # convert integers to dummy variables (i.e. one hot encoded)
        result = np_utils.to_categorical(encoded_Y)

        return source_data, result

    def __get_elements_from_file(self, text_path = "test.txt") :
        parser = Parser(Parser.CHARACTER)
        elements = parser.read(getAbsolutePath(text_path),True)
        return elements

    def __get_elements_from_text(self,text):
        parser = Parser(Parser.CHARACTER)
        elements = parser.read(text,False)
        return elements

