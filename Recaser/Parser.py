class Parser:

    def __init__(self) :
        """Constructor"""

    def read(self):
        with open('test.txt', 'r') as f :
            for line in f :
                for word in line.split() :
                    print(word)