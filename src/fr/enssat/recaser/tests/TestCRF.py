from src.fr.enssat.recaser.CRF.WordCRFRecaser import WordCRFRecaser


if __name__ == "__main__" :

    crfRecaser = WordCRFRecaser()
    crfRecaser.initModel()
    crfRecaser.testModel()
