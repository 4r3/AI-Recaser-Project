from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.CRF.WordCRFRecaser import CRFRecaser


def getAbsolutePath(file_name) :
    """Compute the absolute path of the file if present in the 'resources' directory"""
    import os
    basepath = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", file_name))

if __name__ == "__main__" :
    parser = Parser()
    elements_train = parser.read(getAbsolutePath("set_1/learn_set.txt"))
    elements_test = parser.read(getAbsolutePath("set_1/validate_set.txt"))

    #for element in elements_test :
        #print(element)

    crfRecaser = CRFRecaser()

    crfRecaser.validation(elements_train, elements_test)

    #print(crfRecaser.generateText(elements_train, elements_test)) 