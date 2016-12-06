from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.SentenceElement import SentenceElement


class Parser(object) :
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

    def read(self, content, isFile = False) :
        if self.mode == self.CHARACTER and isFile == False :
            return self.__readAsCharText(content)
        elif self.mode == self.CHARACTER and isFile == True :
            return self.__readAsCharFile(content)
        elif self.mode == self.WORD_NLTK and isFile == False :
            return self.__readAsWordNLTKText(content)
        elif self.mode == self.WORD_NLTK and isFile == True :
            return self.__readAsWordNLTKFile(content)

    # ===============
    # PRIVATE METHODS
    # ===============

    def __readAsWordNLTKFile(self, file_name) :
        from nltk import word_tokenize
        from nltk import pos_tag
        elements = []

        text = ""
        with open(file_name, 'r') as file :
            for line in file :
                text += line
        tokens_tags = pos_tag(word_tokenize(text))

        for token in tokens_tags :
            if token[0].isupper() :
                operation = RecaserOperation.FULL_UPPER
            elif token[0][0].isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING

            element = SentenceElement(token[0].lower(), token[1], operation)

            for existing in elements :
                if existing.value[0] == token[0].lower() :
                    element.id = existing.id
            elements.append(element)

        return elements

    def __readAsWordNLTKText(self, text) :
        from nltk import word_tokenize
        from nltk import pos_tag
        elements = []

        tokens_tags = pos_tag(word_tokenize(text))

        for token in tokens_tags :
            if token[0].isupper() :
                operation = RecaserOperation.FULL_UPPER
            elif token[0][0].isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING

            element = SentenceElement(token[0].lower(), token[1], operation)

            for existing in elements :
                if existing.value[0] == token[0].lower() :
                    element.id = existing.id
            elements.append(element)

        return elements

    def __readAsCharFile(self, file_name) :
        with open(file_name, 'r') as file :
            elements = []
            for line in file :
                tmp = list(line)
                for item in tmp :
                    if item.isupper() :
                        operation = RecaserOperation.START_UPPER
                    else :
                        operation = RecaserOperation.NOTHING

                    element = SentenceElement(item, None, operation)
                    elements.append(element)
        return elements

    def __readAsCharText(self, text) :
        elements = []
        for item in text :
            if item.isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING

            element = SentenceElement(item, None, operation)
            elements.append(element)

        return elements
