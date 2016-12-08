from src.fr.enssat.recaser.CRF.CRFRecaser import CRFRecaser
from src.fr.enssat.recaser.validation.Validation import Validation

if __name__ == "__main__" :

    crfRecaser = CRFRecaser(CRFRecaser.WORD)
    validation = Validation(CRFRecaser.WORD)
    crfRecaser.initModel()
    [correct, predict] = crfRecaser.predictAndTest()
    print(validation.confusionMatrix(correct, predict, True))
    print(validation.classificationReport(correct, predict))
