import time

import pycrfsuite
from sklearn.metrics import confusion_matrix

from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.utils.TextLoader import TextLoader
from src.fr.enssat.recaser.validation.Validation import Validation


class WordCRFRecaser(object) :
    prediction = []
    correct = []

    #
    #   Public methods
    #

    def prepare_training(self, trainingData) :
        X_train = self.sent2features(trainingData)
        Y_train = self.sent2labels(trainingData)
        return (X_train, Y_train)

    def prepare_test(self, testData) :
        X_test = self.sent2features(testData)
        Y_test = self.sent2labels(testData)
        return (X_test, Y_test)

    # Features pour le Recaser
    def word2features(self, sent, i) :
        word = str(sent[i].value)
        tag = sent[i].tag
        features = [
            'bias',
            'word.lower=' + word,
            'tag=' + tag,
        ]
        if i > 0 :
            word1 = str(sent[i - 1].value)
            tag1 = sent[i - 1].tag
            features.extend([
                '-1:word.lower=' + word1,
                '-1:tag=' + tag1,
            ])
        else :
            features.append('BOS')

        if i < len(sent) - 1 :
            word1 = str(sent[i + 1].value)
            tag1 = sent[i + 1].tag
            features.extend([
                '+1:word.lower=' + word1,
                '+1:tag=' + tag1,
            ])
        else :
            features.append('EOS')

        return features

    def train(self, X_train, Y_train) :
        trainer = pycrfsuite.Trainer(algorithm = 'lbfgs', verbose = False)
        # print(trainer.params())

        trainer.append(X_train, Y_train)

        trainer.set_params({
            # 'c1': 1.0,   # coefficient for L1 penalty
            # 'c2': 1e-3,  # coefficient for L2 penalty
            # 'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions' : True,
            'feature.possible_states' : True
        })
        trainer.train('../../../../../resources/models/trainingModelWord.crfsuite')

    def test(self, X_test) :
        tagger = pycrfsuite.Tagger()
        tagger.open('../../../../../resources/models/trainingModelWord.crfsuite')
        return tagger.tag(X_test)

    def generateText(self, elements_train, elements_test) :
        [X_train, Y_train] = self.prepare_training(elements_train)
        [X_test, self.correct] = self.prepare_test(elements_test)
        value_test = self.sent2tokens(elements_test)

        self.train(X_train, Y_train)
        self.prediction = self.test(X_test)

        result = ""

        for value, type in zip(value_test, self.prediction) :
            # print(value)
            # print(type)
            if type == "0" :
                result += " " + value
            elif type == "1" :
                result += " " + value[0].upper() + value[1 :]
            elif type == "2" :
                result += " " + value.upper()

        return result[1 :]

    def initModel(self) :
        elements_train = self.loadFile("corpus_1/corpus")
        [X_train, Y_train] = self.prepare_training(elements_train)

        elements_train = self.loadFile("corpus_2/corpus")
        [X_train2, Y_train2] = self.prepare_training(elements_train)
        X_train.extend(X_train2)
        Y_train.extend(Y_train2)

        start_time = time.time()
        self.train(X_train, Y_train)
        print('Model compiled in {0} seconds'.format(time.time() - start_time))

    def testModel(self) :
        elements_test = self.loadFile("corpus_6/corpus")
        [X_test, self.correct] = self.prepare_test(elements_test)

        print("Start prediction")
        start_time = time.time()
        self.prediction = self.test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        validation = Validation(1)
        print(validation.confusionMatrix(self.correct, self.prediction))
        print(validation.confusionMatrix(self.correct, self.prediction, True))
        print(validation.classificationReport(self.correct, self.prediction))

    def predictFileAndTest(self, file) :
        elements_test = self.loadFile(file)
        print("Loaded")
        [X_test, self.correct] = self.prepare_test(elements_test)

        print("Start prediction")
        start_time = time.time()
        self.prediction = self.test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        confusionMatrix = confusion_matrix(self.correct, self.prediction)
        print(confusionMatrix)

    def predictFileAndTest(self, sentence) :
        elements_test = self.loadSentence(sentence)
        print("Loaded")
        [X_test, self.correct] = self.prepare_test(elements_test)

        print("Start prediction")
        start_time = time.time()
        self.prediction = self.test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        confusionMatrix = confusion_matrix(self.correct, self.prediction)
        print(confusionMatrix)

    def predictFile(self, file) :
        elements_test = self.loadFile(file)
        print("Loaded")
        X_test = self.sent2features(elements_test)

        print("Start prediction")
        start_time = time.time()
        self.prediction = self.test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        return self.prediction

    def predictSentence(self, sentence) :
        elements_test = self.loadSentence(sentence)
        print("Loaded")
        X_test = self.sent2features(elements_test)

        print("Start prediction")
        start_time = time.time()
        self.prediction = self.test(X_test)
        print('Model tested in {0} seconds'.format(time.time() - start_time))

        return self.prediction

    #
    #   Private methods
    #

    # Création des features
    def sent2features(self, sent) :
        return [self.word2features(sent, i) for i in range(len(sent))]

    # Création des labels
    def sent2labels(self, sent) :
        return [str(message.operation) for message in sent]

    # Création des valeurs test
    def sent2tokens(self, sent) :
        return [message.value for message in sent]

    # Charger les fichiers
    def loadFile(self, filePath) :
        loader = TextLoader()
        text = loader.get_text(filePath, False)
        parser = Parser(Parser.MODE_WORD)
        return parser.read(text, False)
