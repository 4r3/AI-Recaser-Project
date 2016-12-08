from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


class Validation(object) :
    CHAR = 0
    WORD = 1

    def __init__(self, mode = 0) :
        self.mode = mode

    def confusionMatrix(self, y_correct, y_predict, normalize = False) :
        """Generate the confusion matrix,
            y_correct is the expected result and y_predict is the prediction
            normalize at True normalizes the confusion matrix"""
        if self.mode == 0 :
            labels = ['0', '1']
        elif self.mode == 1 :
            labels = ['0', '1', '2']
        confusionMatrix = confusion_matrix(y_correct, y_predict, labels)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if normalize :
            confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis = 1)[:, np.newaxis]
            plt.title('Normalized confusion matrix')
        else :
            plt.title('Confusion matrix')
        cax = ax.matshow(confusionMatrix)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.ylabel('Correct label')
        plt.xlabel('Prediction label')
        plt.show()
        return confusionMatrix

    def classificationReport(self, y_correct, y_predict) :
        """Generate the precision, recall and f1-score,
            y_correct is the expected result and y_predict is the prediction"""
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_correct)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_predict)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key = lambda tag : tag.split('-', 1)[: :-1])
        class_indices = {cls : idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )
