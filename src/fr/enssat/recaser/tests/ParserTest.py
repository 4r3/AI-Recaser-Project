# PyCharm maye indicates an error, but it's ok ;)
from src.fr.enssat.recaser.parser.Parser import Parser

def getAbsolutePath(file_name) :
    """Compute the absolute path of the file if present in the 'resources' directory"""
    import os
    basepath = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", file_name))


if __name__ == "__main__":
    parser = Parser(Parser.WORD)
    elements = parser.read(getAbsolutePath("test.txt"))

    for element in elements:
        print(element)
