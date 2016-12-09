from src.fr.enssat.recaser.CRF.CRFRecaser import CRFRecaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.validation.Validation import Validation


class Tester(object) :
    # ==============
    # PUBLIC METHODS
    # ==============

    def test(self, correct, predict, method) :
        if method == RecaserMethod.DNN_CHAR :
            return 0

        elif method == RecaserMethod.DNN_WORD :
            return 0

        elif method == RecaserMethod.DNN_NAEN :
            return 0

        elif method == RecaserMethod.CRF_CHAR :
            validation = Validation(Validation.CHAR)
            print("------------------------------ VALIDATION...")
            print("------------------ Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, False))
            print("------------------ Normalized Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, True))
            print("------------------ Classification Report...")
            print(validation.classificationReport(correct, predict))

        elif method == RecaserMethod.CRF_WORD :
            validation = Validation(Validation.WORD)
            print("------------------------------ VALIDATION...")
            print("------------------ Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, False))
            print("------------------ Normalized Confusion Matrix...")
            print(validation.confusionMatrix(correct, predict, True))
            print("------------------ Classification Report...")
            print(validation.classificationReport(correct, predict))