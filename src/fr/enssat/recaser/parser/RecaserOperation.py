class RecaserOperation(enumerate) :
    """
        Enumeration representing the operation done to a given SentenceElement during the lowercase.
        NOTHING: the value was already in lower case;
        START_UPPER: the first letter of the value was upper case (no matter about the next
        FULL_UPPER: the whole value was upper case
    """

    NOTHING = 0
    START_UPPER = 1
    FULL_UPPER = 2
