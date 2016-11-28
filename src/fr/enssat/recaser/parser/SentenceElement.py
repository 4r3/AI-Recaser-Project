from itertools import count


class SentenceElement(object):
    last_id = count(0)  # Instance counter

    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, value, operation):
        self.id = next(self.last_id)
        self.value = value
        self.operation = operation  # Operation to do

    # =========
    # UTILITIES
    # =========

    def __str__(self):
        if self.value.__class__.__name__ in ('tuple'):
            return "SentenceElement[ id = " + str(self.id) + " | value = " + self.value[0] + " | tag = " + self.value[1] + " | operation = " + str(self.operation) + " ]"
        else: # CHAR MODE
            return "SentenceElement[ id = " + str(self.id) + " | value = " + self.value + " | operation = " + str(self.operation) + " ]"

