class SentenceElement(object) :
    # ===========
    # CONSTRUCTOR
    # ===========

    def __init__(self, value, tag, operation, tag_bin, tag_bin_index) :
        """Creates a new SentenceElement with no id."""
        self.id = None
        self.value = value
        self.tag = tag
        self.operation = operation  # See RecaserOperation.py
        self.tag_bin_array = tag_bin
        self.tag_bin_index = tag_bin_index

    # =========
    # UTILITIES
    # =========

    def __str__(self) :
        return "SentenceElement[ id = " + str(self.id) + " | value = " + self.value + " | tag = " + str(self.tag) + " | operation = " + str(self.operation) + " | tag binaire index : " + str(self.tag_bin_index) + " ]"
