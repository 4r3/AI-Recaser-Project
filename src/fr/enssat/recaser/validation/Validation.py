import itertools
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
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        thresh = confusionMatrix.max() / 2.
        for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
            plt.text(j, i, round(confusionMatrix[i, j], 2),
                     horizontalalignment="center",
                     color="white" if confusionMatrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()

        return confusionMatrix

    def classificationReport(self, y_correct, y_predict) :
        """Generate the precision, recall and f1-score,
            y_correct is the expected result and y_predict is the prediction"""

        y_correct = [str(item) for item in y_correct]
        y_predict = [str(item) for item in y_predict]

        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(y_correct)
        y_pred_combined = lb.transform(y_predict)

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key = lambda tag : tag.split('-', 1)[: :-1])
        class_indices = {cls : idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )
