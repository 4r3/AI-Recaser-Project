import nltk
import numpy as np

from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.parserBis.SentenceElementBis import SentenceElement

# TODO NUMPY MP BOOL POUR TYPE DE TABLEAU ->
# TODO Regarder combien il y a de type de mot (adj,...) et faire faire un tableau de booleen en fonction du type

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

            element = SentenceElement(token[0].lower(), token[1], operation,None,None)

            for existing in elements :
                if existing.value[0] == token[0].lower() :
                    element.id = existing.id
            elements.append(element)

        return elements

    def __readAsWordNLTKText(self, text) :
        from nltk import word_tokenize
        from nltk import pos_tag
        elements = []
        tag_all = [ 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ','JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR','RBS', 'RP', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'SYM', 'VBD', 'VBG','VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

        len_tag_all = (len(tag_all)+1)

        tag_bin=np.zeros((len_tag_all,1),dtype=np.bool)
        tokens_tags = pos_tag(word_tokenize(text))

        for token in tokens_tags :

            print(token[1])
            if(token[1] in tag_all ):
                id = tag_all.index(token[1])
                print(id)
                tag_bin[id] = 1
            else :
                id = len(tag_bin)-1
                tag_bin[id] = 1

            if token[0].isupper() :
                operation = RecaserOperation.FULL_UPPER
            elif token[0][0].isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING
            print('id : '+str(id))
            element = SentenceElement(token[0].lower(), token[1], operation,tag_bin,id)

            tag_bin[id] = None
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

                    element = SentenceElement(item, None, operation,None,None)
                    elements.append(element)
        return elements

    def __readAsCharText(self, text) :
        elements = []
        for item in text :
            if item.isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING

            element = SentenceElement(item, None, operation,None,None)
            elements.append(element)

        return elements
