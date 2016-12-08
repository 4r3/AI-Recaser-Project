from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from itertools import chain

class RecaserMethod(enumerate) :
    DNN_CHAR = 0
    DNN_WORD = 1
    CRF_CHAR = 2
    CRF_WORD = 3

    def confusionMatrix(self, Y_correct, Y_predict):
        confusionMatrix = confusion_matrix(self.correct, self.prediction)
        plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('Correct label')
        plt.xlabel('Prediction label')
        plt.show()

        return confusionMatrix

    def classificationReport(self, Y_correct, Y_predict):
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(Y_correct)))
        y_pred_combined = lb.transform(list(chain.from_iterable(Y_predict)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )