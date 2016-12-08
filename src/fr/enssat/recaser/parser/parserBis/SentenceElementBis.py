from itertools import count
import numpy

class SentenceElement(object) :
    last_id = count(0)  # Instance counter

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, value, tag, operation,binTag) :
        self.id = next(self.last_id)
        self.value = value.lower()
        self.tag = tag
        self.operation = operation  # Operation to do
        self.binTag = binTag



    # =========
    # UTILITIES
    # =========

    def __str__(self) :
        return "SentenceElement[ id = " + str(self.id) + " | value = " + self.value + " | tag = " + str(self.tag) + " | operation = " + str(self.operation) +" | tag binaire = " + str(self.binTag) + " ]"
