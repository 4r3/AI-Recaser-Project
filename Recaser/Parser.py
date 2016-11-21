import re

class Parser:

    def __init__(self) :
        """Empty constructor"""

    def read(self, file_name):
        with open(file_name, 'r') as file :
            for line in file :
                for word in line.split() :
                    print(re.split(',', word))
                    #print(word)