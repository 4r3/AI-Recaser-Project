from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.CRF.WordCRFRecaser import WordCRFRecaser
from src.fr.enssat.recaser.CRF.CharCRFRecaser import CharCRFRecaser


def getAbsolutePath(file_name) :
    """Compute the absolute path of the file if present in the 'resources' directory"""
    import os
    basepath = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", file_name))

if __name__ == "__main__" :

    crfRecaser = WordCRFRecaser()
    crfRecaser.initModel();
    crfRecaser.testModel();
