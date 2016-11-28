import os
import time
import numpy as np
from keras.utils import np_utils
from keras.layers import Embedding, LSTM, Input
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from src.fr.enssat.recaser.tests.ParserTest import getAbsolutePath
from src.fr.enssat.recaser.parser.Parser import Parser


def init_model():
    start_time = time.time()

    print('Compiling Model ... ')
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=200, input_shape=(1,)))

    model.add(LSTM(200, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(20))

    # shape the output
    model.add(Dense(2))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy',])
    print('Model compield in {0} seconds'.format(time.time() - start_time))
    return model


def run_network(data, model, epochs=20, batch=256):
    try:
        start_time = time.time()

        X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        print('Training model...')
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  validation_data=(X_test, y_test), verbose=2, shuffle=False)

        print("Training duration : {0}".format(time.time() - start_time))

        return model
    except KeyboardInterrupt:
        print(' KeyboardInterrupt')
        return model


def load_model():
    model_path = "model.json"

    if os.path.isfile(model_path):
        with open(model_path, 'r') as model_file:
            model_json = model_file.read()
        model = model_from_json(model_json)
        rms = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    else:
        model = init_model()
        with open(model_path, 'w') as model_file:
            model_file.write(model.to_json())

    if os.path.isfile('weight.h5'):
        model.load_weights('weight.h5', by_name=True)

    return model


def format_text(source, result):
    for ndx, member in enumerate(source):
        source[ndx] = ord(source[ndx])

    source = np.array(source)
    source = np.reshape(source, (len(source), 1))

    encoder = LabelEncoder()
    encoder.fit(result)
    encoded_Y = encoder.transform(result)
    # convert integers to dummy variables (i.e. one hot encoded)
    result = np_utils.to_categorical(encoded_Y)

    return source, result


def get_formatted_text(text_path="test.txt"):
    parser = Parser(Parser.CHARACTER)
    elements = parser.read(getAbsolutePath(text_path))

    source = []
    result = []
    for element in elements:
        source.append(element.value)
        result.append(element.operation)

    print(source)
    print(result)
    return [source, result]

def learn_and_return(ressources_path="set_1"):
    source, result = get_formatted_text(ressources_path + "/learn_set.txt")
    learn_text, learn_result = format_text(source,result)
    source, result = get_formatted_text(ressources_path + "/validate_set.txt")
    test_text, test_result = format_text(source,result)

    data = [learn_text, test_text, learn_result, test_result]

    model = init_model()

    model = run_network(data,model,batch=256, epochs=2)

    y = model.predict(test_text)

    y_classes = np_utils.categorical_probas_to_classes(y)

    return y_classes

print(learn_and_return())
# model.save("weight.h5")
