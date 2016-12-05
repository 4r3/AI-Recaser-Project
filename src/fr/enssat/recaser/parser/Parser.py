import re as regex

from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.SentenceElement import SentenceElement


class Parser(object):
    WORD = "word"
    WORD_NLTK = "word_nltk"
    CHARACTER = "char"

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, mode="word"):
        self.mode = mode

    # ================
    # PUBLIC FUNCTIONS
    # ================

    def read(self, file_name, isFile=False):
        if self.mode == self.WORD:
            return self.__readAsWord(file_name)
        elif self.mode == self.CHARACTER:
            return self.__readAsChar(file_name)
        else:
            return self.__readAsWordNLTK(file_name)

    # ===============
    # PRIVATE METHODS
    # ===============

    #@deprecated
    def __readAsWord(self, file_name):
        from string import punctuation

        pattern = regex.compile(r'\w+|[{}]'.format(regex.escape(punctuation)))
        elements = []
        with open(file_name, 'r') as file:
            for line in file:
                parts = pattern.findall(line)
                for part in parts:
                    if part.isupper():
                        operation = RecaserOperation.FULL_UPPER
                    elif part[0].isupper():
                        operation = RecaserOperation.START_UPPER
                    else:
                        operation = RecaserOperation.NOTHING

                    element = SentenceElement(part, operation)
                    elements.append(element)

        return elements

    def __readAsWordNLTK(self, file_name):
        from nltk import word_tokenize
        from nltk import pos_tag
        elements = []

        text = ""
        with open(file_name, 'r') as file:
            for line in file:
                text += line
        tokens_tags = pos_tag(word_tokenize(text))

        for token in tokens_tags:
            if token[0].isupper():
                operation = RecaserOperation.FULL_UPPER
            elif token[0][0].isupper():
                operation = RecaserOperation.START_UPPER
            else:
                operation = RecaserOperation.NOTHING

            element = SentenceElement(token[0].lower(), token[1], operation)

            for existing in elements:
                if existing.value[0] == token[0].lower():
                    element.id = existing.id
            elements.append(element)

        return elements

    def __readAsChar(self, file_name):
        with open(file_name, 'r') as file:
            elements = []
            for line in file:
                tmp = list(line)
                for item in tmp:
                    if item.isupper():
                        operation = RecaserOperation.START_UPPER
                    else:
                        operation = RecaserOperation.NOTHING

                    element = SentenceElement(item, operation)
                    elements.append(element)
        return elements

