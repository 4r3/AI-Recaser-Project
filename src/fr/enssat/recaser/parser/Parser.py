from itertools import count

import itertools
from nltk import pos_tag, word_tokenize
from nltk.stem.snowball import EnglishStemmer

from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.SentenceElement import SentenceElement


class Parser(object) :
    WORD = 1
    CHARACTER = 2

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, mode, stemmer = EnglishStemmer()) :
        self.mode = mode
        self.stemmer = stemmer

    # ================
    # PUBLIC FUNCTIONS
    # ================

    def read(self, content, stemming=False) :
        if self.mode == self.CHARACTER:
            elements = self.__readAsChar(content)
        elif self.mode == self.WORD:
            elements = self.__readAsWord(content,stemming)
        else:
            raise Exception("Invalid mode")

        return elements

    # ===============
    # PRIVATE METHODS
    # ===============

    def __readAsWord(self, text, stemming=False) :
        elements = []

        ll = [[word_tokenize(w), ' '] for w in text.split()]
        tokens_tags = pos_tag(list(itertools.chain(*list(itertools.chain(*ll)))))

        for token in tokens_tags:
            if token[0].isupper():
                operation = RecaserOperation.FULL_UPPER
            elif token[0][0].isupper() :
                operation = RecaserOperation.START_UPPER
            else :
                operation = RecaserOperation.NOTHING

            if stemming:
                value = self.stemmer.stem(token[0]) # Stem also apply lower() function
            else:
                value = token[0].lower()

            if value == " ":
                tag = " "
            else:
                tag = token[1]

            element = SentenceElement(value, tag, operation)

            for existing in elements :
                if existing.value == element.value:
                    element.id = existing.id
            elements.append(element)

        return elements

    def __readAsChar(self, text) :
        # Parse as words
        elements = self.__readAsWord(text)
        SentenceElement.last_id = count(0) # Ugly temporary fix

        # Create element for each char of each word
        new_elements = []
        for element in elements:
            for letter in element.value:
                if letter.isupper() :
                    operation = RecaserOperation.START_UPPER
                else :
                    operation = RecaserOperation.NOTHING
                new_element = SentenceElement(letter, element.tag, operation)

                for existing in new_elements :
                    if existing.value[0].lower() == new_element.value.lower() :
                        new_element.id = existing.id
                new_elements.append(new_element)

        return new_elements
