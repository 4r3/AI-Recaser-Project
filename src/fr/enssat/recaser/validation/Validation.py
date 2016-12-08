from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from itertools import chain


class Validation(object):
    def confusionMatrix(self, y_correct, y_predict):
        confusionMatrix = confusion_matrix(y_correct, y_predict)
        plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('Correct label')
        plt.xlabel('Prediction label')
        plt.show()

        return confusionMatrix

    def classificationReport(self, y_correct, y_predict):
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_correct)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_predict)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )