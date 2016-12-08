from itertools import count
import numpy

class SentenceElement(object) :
    last_id = count(0)  # Instance counter

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, value, tag, operation,tag_bin_array, tag_bin_index) :
        self.id = next(self.last_id)
        self.value = value.lower()
        self.tag = tag
        self.operation = operation  # Operation to do
        self.tag_bin_array = tag_bin_array
        self.tag_bin_index = tag_bin_index



    # =========
    # UTILITIES
    # =========

    def __str__(self) :
        #if(self.tag_bin_array != None and self.tag_bin_index!=None) :
            return "SentenceElement[ id = " + str(self.id) + " | value = " + self.value + " | tag = " + str(self.tag) + " | operation = " + str(self.operation) +" | tag binaire array= "+ str(self.tag_bin_array) + " |tag binaire index : "+str(self.tag_bin_index) +" ]"
        #else :
         #   return "SentenceElement[ id = " + str(self.id) + " | value = " + self.value + " | tag = " + str(self.tag) + " | operation = " + str(self.operation) + " ]"
