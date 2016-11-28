import re as regex

from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.SentenceElement import SentenceElement


class Parser(object) :
    WORD = "word"
    WORD_NLTK = "word_nltk"
    CHARACTER = "char"

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, mode = "word") :
        self.mode = mode

    # ================
    # PUBLIC FUNCTIONS
    # ================

    def read(self, file_name) :
        if self.mode == self.WORD :
            return self.__readAsWord(file_name)
        elif self.mode == self.CHARACTER :
            return self.__readAsChar(file_name)
        else:
            return self.__readAsWordNLTK(file_name)

    # ===============
    # PRIVATE METHODS
    # ===============

    def __readAsWord(self, file_name) :
        from string import punctuation

        pattern = regex.compile(r'\w+|[{}]'.format(regex.escape(punctuation)))
        elements = []
        with open(file_name, 'r') as file :
            for line in file :
                parts = pattern.findall(line)
                for part in parts :
                    if part.isupper() :
                        operation = RecaserOperation.FULL_UPPER
                    elif part[0].isupper() :
                        operation = RecaserOperation.START_UPPER
                    else :
                        operation = RecaserOperation.NOTHING

                    element = SentenceElement(part, operation)
                    elements.append(element)

        return elements

    def __readAsWordNLTK(self, file_name) :
        from nltk import word_tokenize
        from nltk import pos_tag

        text = ""
        with open(file_name, 'r') as file:
            for line in file:
                text += line

        tokens = word_tokenize(text)
        return pos_tag(tokens)

    def __readAsChar(self, file_name):
        with open(file_name, 'r') as file:
            elements = []
            for line in file:
                tmp = list(line)
                for item in tmp:
                    if item.isupper():
                        operation = RecaserOperation.START_UPPER
                    else :
                        operation = RecaserOperation.NOTHING

                    element = SentenceElement(item, operation)
                    elements.append(element)
        return elements