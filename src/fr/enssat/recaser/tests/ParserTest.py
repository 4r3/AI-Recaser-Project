# PyCharm maye indicates an error, but it's ok ;)
from src.fr.enssat.recaser.parser.Parser import Parser

def getAbsolutePath(param) :
    """Compute the absolute path of the file if present in the 'resources' directory"""
    import os
    basepath = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", "test.txt"))


if __name__ == "__main__":
    parser = Parser()
    parser.read(getAbsolutePath("test.txt"))
