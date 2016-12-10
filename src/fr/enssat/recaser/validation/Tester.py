from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.validation.Validation import Validation


class Tester(object):
    # ==============
    # PUBLIC METHODS
    # ==============

    def test(self, correct, predict, method) :
        if method == Validation.CHAR :
            validation = Validation(Validation.CHAR)
            print("------------------------------ VALIDATION...")
            print("------------------ Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, False))
            print("------------------ Normalized Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, True))
            print("------------------ Classification Report...")
            print(validation.classificationReport(correct, predict))

        elif method == Validation.WORD :
            validation = Validation(Validation.WORD)
            print("------------------------------ VALIDATION...")
            print("------------------ Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, False))
            print("------------------ Normalized Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, True))
            print("------------------ Classification Report...")
            print(validation.classificationReport(correct, predict))