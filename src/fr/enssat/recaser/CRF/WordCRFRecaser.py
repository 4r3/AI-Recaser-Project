from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import matplotlib.pyplot as plt

class CRFRecaser(object):
    prediction = []
    correct = []

    #
    #   Public methods
    #

    def prepare_training(self, trainingData):
        X_train = self.sent2features(trainingData)
        Y_train = self.sent2labels(trainingData)
        return (X_train, Y_train)

    def prepare_test(self, testData):
        X_test = self.sent2features(testData)
        Y_test = self.sent2labels(testData)
        return (X_test, Y_test)

    # Features pour le Recaser
    def word2features(self, sent, i):
        word = str(sent[i].value)
        #postag = sent[i][1]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            #'postag=' + postag,
        ]
        if i > 0:
            word1 = str(sent[i-1].value)
            #postag1 = sent[i-1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                #'-1:postag=' + postag1,
            ])
        else:
            features.append('BOS')

        if i < len(sent)-1:
            word1 = str(sent[i+1].value )
            #postag1 = sent[i+1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                #'+1:postag=' + postag1,
            ])
        else:
            features.append('EOS')

        return features

    def train(self, X_train, Y_train):
        trainer = pycrfsuite.Trainer(verbose=False)

        trainer.append(X_train, Y_train)

        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train('trainingModel.crfsuite')

    def test(self, X_test):
        tagger = pycrfsuite.Tagger()
        tagger.open('trainingModel.crfsuite')

        #print("Predicted:", ' '.join(tagger.tag(X_test)))
        #print("Correct:  ", ' '.join(Y_test))
        return tagger.tag(X_test)

    def bio_classification_report(self, y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.

        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )

    def validation(self, elements_train, elements_test):
        [X_train, Y_train] = self.prepare_training(elements_train)
        [X_test, self.correct] = self.prepare_test(elements_test)

        self.train(X_train, Y_train)
        self.prediction = self.test(X_test)
        confusionMatrix = confusion_matrix(self.correct, self.prediction)
        plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('Correct label')
        plt.xlabel('Prediction label')
        plt.show()

        return self.bio_classification_report(self.correct, self.prediction)

    def generateText(self, elements_train, elements_test):
        [X_train, Y_train] = self.prepare_training(elements_train)
        [X_test, self.correct] = self.prepare_test(elements_test)
        value_test = self.sent2tokens(elements_test)

        self.train(X_train, Y_train)
        self.prediction = self.test(X_test)

        result = ""

        for value, type in zip(value_test, self.prediction):
            #print(value)
            #print(type)
            if type == "0":
                result += " " + value
            elif type == "1":
                result += " " + value[0].upper() + value[1:]
            elif type == "2":
                result += " " + value.upper()

        return result[1:]


    #
    #   Private methods
    #

    #Création des features
    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    #Création des labels
    def sent2labels(self, sent):
        return [str(message.operation) for message in sent]

    #Création des valeurs test
    def sent2tokens(self, sent):
        return [message.value for message in sent]