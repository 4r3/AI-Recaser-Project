import time
import pycrfsuite
from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.utils.TextLoader import TextLoader


class CRFRecaser(object):
    CHAR = 0
    WORD = 1
    prediction = []
    correct = []

    def __init__(self, mode=0):
        self.mode = mode

    #
    #   Public methods
    #

    def initModel(self, file='corpus_1/corpus'):
        elements_train = self.__loadFile(file)
        [X_train, Y_train] = self.__prepare_training(elements_train)

        start_time = time.time()
        self.__train(X_train, Y_train)
        print('Model compiled in {0} seconds'.format(time.time() - start_time))

    def predictAndTest(self, file='corpus_2/corpus'):
        elements_test = self.__loadFile(file)
        [X_test, self.correct] = self.__prepare_test(elements_test)

        start_time = time.time()
        self.prediction = self.__test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        return [self.correct, self.prediction]

    def predict(self, file='corpus_2/corpus'):
        elements_test = self.__loadFile(file)
        X_test = self.__sent2features(elements_test)

        start_time = time.time()
        self.prediction = self.__test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        return self.prediction

    def getLastResult(self):
        return self.prediction


    #
    #   Private methods
    #

    # Création des features
    def __sent2features(self, sent):
        return [self.__word2features(sent, i) for i in range(len(sent))]

    # Création des labels
    def __sent2labels(self, sent):
        return [str(message.operation) for message in sent]

    # Charger les fichiers
    def __loadFile(self, filePath):
        loader = TextLoader()
        text = loader.getText(filePath, False)
        if self.mode == self.WORD:
            parser = Parser(Parser.WORD)
        else:
            parser = Parser(Parser.CHARACTER)
        return parser.read(text, False)

    # Features pour le Recaser
    def __word2features(self, sent, i):
        word = str(sent[i].value)
        tag = sent[i].tag
        features = [
            'bias',
            'word.lower=' + word,
            'tag=' + tag,
        ]
        if i > 0:
            word1 = str(sent[i - 1].value)
            tag1 = sent[i - 1].tag
            features.extend([
                '-1:word.lower=' + word1,
                '-1:tag=' + tag1,
            ])
            if i > 1:
                word1 = str(sent[i - 2].value)
                tag1 = sent[i - 2].tag
                features.extend([
                    '-2:word.lower=' + word1,
                    '-2:tag=' + tag1,
                ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = str(sent[i + 1].value)
            tag1 = sent[i + 1].tag
            features.extend([
                '+1:word.lower=' + word1,
                '+1:tag=' + tag1,
            ])
            if i < len(sent) - 2:
                word1 = str(sent[i + 2].value)
                tag1 = sent[i + 2].tag
                features.extend([
                    '+2:word.lower=' + word1,
                    '+2:tag=' + tag1,
                ])
        else:
            features.append('EOS')

        return features

    def __prepare_training(self, trainingData):
        X_train = self.__sent2features(trainingData)
        Y_train = self.__sent2labels(trainingData)
        return (X_train, Y_train)

    def __prepare_test(self, testData):
        X_test = self.__sent2features(testData)
        Y_test = self.__sent2labels(testData)
        return (X_test, Y_test)

    def __train(self, X_train, Y_train):
        trainer = pycrfsuite.Trainer(algorithm='lbfgs', verbose=False)
        trainer.append(X_train, Y_train)
        trainer.set_params({
            'feature.possible_states': True
        })
        trainer.train('../../../../../resources/models/trainingModelWord.crfsuite')

    def __test(self, X_test):
        tagger = pycrfsuite.Tagger()
        tagger.open('../../../../../resources/models/trainingModelWord.crfsuite')
        return tagger.tag(X_test)