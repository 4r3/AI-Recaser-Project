from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.CRF.WordCRFRecaser import WordCRFRecaser
from src.fr.enssat.recaser.CRF.CharCRFRecaser import CharCRFRecaser


def getAbsolutePath(file_name) :
    """Compute the absolute path of the file if present in the 'resources' directory"""
    import os
    basepath = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", file_name))

if __name__ == "__main__" :
    #parser = Parser(Parser.WORD_NLTK)
    #elements_train = parser.read(getAbsolutePath("set_1/learn_set.txt"), True)
    #elements_test = parser.read(getAbsolutePath("set_2/alice_underground_1"), True)

    #crfRecaser = WordCRFRecaser()

    #crfRecaser.validation(elements_train, elements_test)


    parser = Parser(Parser.CHARACTER)
    elements_train = parser.read(getAbsolutePath("set_1/learn_set.txt"), True)
    print(elements_train[0])
    elements_test = parser.read(getAbsolutePath("set_2/alice_underground_1"), True)

    crfRecaser = CharCRFRecaser()

    crfRecaser.validation(elements_train, elements_test)

    #print(crfRecaser.generateText(elements_train, elements_test))