import re

class Parser :
    def __init__(self) :
        """Empty constructor"""

    def read(self, file_name) :
        from string import punctuation

        pattern = re.compile(r'\w+|[{}]'.format(re.escape(punctuation)))
        with open(file_name, 'r') as file :
            for line in file:
                print(pattern.findall(line))

