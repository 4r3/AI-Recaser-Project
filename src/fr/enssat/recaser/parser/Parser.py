import itertools
from itertools import count

import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.stem.snowball import EnglishStemmer

from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.SentenceElement import SentenceElement


class Parser(object) :
    WORD = 1
    CHARACTER = 2

    TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'SYM', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', ' ']

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, mode, stemmer = EnglishStemmer()) :
        self.mode = mode
        self.stemmer = stemmer

    # ================
    # PUBLIC FUNCTIONS
    # ================

    def read(self, content, stemming = False) :
        if self.mode == self.CHARACTER :
            elements = self.__readAsChar(content)
        elif self.mode == self.WORD :
            elements = self.__readAsWord(content, stemming) #FIXME: setemming or force to false ?? (voir antoine)
        else :
            raise Exception("Invalid mode")

        return elements

    # ===============
    # PRIVATE METHODS
    # ===============

    def __readAsWord(self, text, stemming = False) :
        elements = []
        tag_bin = np.zeros((len(self.TAGS) + 1, 1), dtype = np.bool)

        ll = [[word_tokenize(w), ' '] for w in text.split()] # TODO: rename
        tokens_tags = pos_tag(list(itertools.chain(*list(itertools.chain(*ll)))))

        for token in tokens_tags :
            if token[0].isupper() :
                operation = RecaserOperation.FULL_UPPER
            elif token[0][0].isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING

            if stemming :
                value = self.stemmer.stem(token[0])  # Stem also apply lower() function
            else :
                value = token[0].lower()

            if value == " " :
                tag = " "
            else :
                tag = token[1]

            if (tag in self.TAGS) :
                tag_bin_index = self.TAGS.index(tag)
                tag_bin[tag_bin_index] = 1
            else :
                tag_bin_index = len(tag_bin) - 1
                tag_bin[tag_bin_index] = 1

            element = SentenceElement(value, tag, operation,tag_bin,tag_bin_index)

            tag_bin[tag_bin_index] = None
            for existing in elements :
                if existing.value == element.value :
                    element.id = existing.id
            elements.append(element)

        return elements

    def __readAsChar(self, text) :
        # Parse as words
        elements = self.__readAsWord(text)
        SentenceElement.last_id = count(0)  # Ugly temporary fix

        tag_bin = np.zeros((len(self.TAGS) + 1, 1), dtype = np.bool)

        # Create element for each char of each word
        new_elements = []
        for element in elements :
            for letter in element.value :
                if letter.isupper() :
                    operation = RecaserOperation.START_UPPER
                else :
                    operation = RecaserOperation.NOTHING

                if (element.tag in self.TAGS) :
                    tag_bin_index = self.TAGS.index(element.tag)
                    tag_bin[tag_bin_index] = 1
                else :
                    tag_bin_index = len(tag_bin) - 1
                    tag_bin[tag_bin_index] = 1

                new_element = SentenceElement(letter, element.tag, operation, tag_bin, tag_bin_index)

                tag_bin[tag_bin_index] = None
                for existing in new_elements :
                    if existing.value[0].lower() == new_element.value.lower() :
                        new_element.id = existing.id
                new_elements.append(new_element)

        return new_elements
