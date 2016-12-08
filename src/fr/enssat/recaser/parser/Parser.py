import itertools

import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.stem.snowball import EnglishStemmer

from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.parser.SentenceElement import SentenceElement
from src.fr.enssat.recaser.utils.DictionaryLoader import DictionaryLoader


class Parser(object) :
    MODE_WORD = 1
    MODE_CHARACTER = 2

    NLTK_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
            'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
            'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'S', 'SBAR', 'SBARQ',
            'SINV', 'SQ', 'SYM', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WDT', 'WP', 'WP$', 'WRB', ',', '.', ' ']

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, mode, stemmer = EnglishStemmer()) :
        """Creates a new Parser with the given mode and the given stemmer. If no stemmer provided, the default 'EnglishStemmer' will be used."""
        self.mode = mode
        self.stemmer = stemmer
        if self.mode == Parser.MODE_WORD:
            self.dictionary = DictionaryLoader.load_dictionary("default_dictionary_word.yaml")
        else:
            self.dictionary = DictionaryLoader.load_dictionary("default_dictionary_char.yaml")

    # ================
    # PUBLIC FUNCTIONS
    # ================

    def read(self, content, stemming = False) :
        """ Convert a text string into a list of 'SentenceElement'. The 'stemming" parameter can be provided only for 'WORD' mode."""
        if self.mode == self.MODE_CHARACTER :
            elements = self.__read_as_char(content)
        elif self.mode == self.MODE_WORD :
            elements = self.__read_as_word(content, stemming)
        else :
            raise Exception("Invalid mode")


        if self.mode == Parser.MODE_WORD:
            DictionaryLoader.save_dictionary(self.dictionary, "default_dictionary_word.yaml")
        else:
            DictionaryLoader.save_dictionary(self.dictionary, "default_dictionary_char.yaml")
        return elements

    # ===============
    # PRIVATE METHODS
    # ===============

    def __read_as_word(self, text, stemming = False, lower = True) :
        """Consider each word as a sentence element. By default all the returned elements are lowercase (lower = True) and no stemming is applied on the given text.
        1. """
        elements = []
        tag_bin = np.zeros((len(self.NLTK_TAGS) + 1, 1), dtype = np.bool) # 'len +1' for any unknown tag

        # Tokenize and keep white spaces
        tmp = [[word_tokenize(w), ' '] for w in text.split()]
        tokens_tags = pos_tag(list(itertools.chain(*list(itertools.chain(*tmp)))))

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
                if lower :
                    value = token[0].lower()
                else :
                    value = token[0]

            if value == " " :
                tag = " "
            else :
                tag = token[1]

            if tag in self.NLTK_TAGS :
                tag_bin_index = self.NLTK_TAGS.index(tag)
                tag_bin[tag_bin_index] = 1
            else :
                tag_bin_index = len(tag_bin) - 1
                tag_bin[tag_bin_index] = 1

            element = SentenceElement(value, tag, operation, tag_bin, tag_bin_index)
            if lower or stemming :
                element.id = self.dictionary.get_id(value)

            tag_bin[tag_bin_index] = None
            elements.append(element)

        elements.pop()
        return elements

    def __read_as_char(self, text) :
        # Parse as words
        elements = self.__read_as_word(text, False, False)  # Don't stem and keep case

        tag_bin = np.zeros((len(self.NLTK_TAGS) + 1, 1), dtype = np.bool)

        # Create element for each char of each word
        new_elements = []
        for element in elements :
            for letter in element.value :
                if letter.isupper() :
                    operation = RecaserOperation.START_UPPER
                else :
                    operation = RecaserOperation.NOTHING

                if element.tag in self.NLTK_TAGS :
                    tag_bin_index = self.NLTK_TAGS.index(element.tag)
                    tag_bin[tag_bin_index] = 1
                else :
                    tag_bin_index = len(tag_bin) - 1
                    tag_bin[tag_bin_index] = 1

                letter = letter.lower()

                new_element = SentenceElement(letter, element.tag, operation, tag_bin, tag_bin_index)
                new_element.id = self.dictionary.get_id(letter)

                tag_bin[tag_bin_index] = None
                new_elements.append(new_element)

        return new_elements
