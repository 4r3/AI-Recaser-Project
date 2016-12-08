from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.validation.Restorator import Restorator


def getAbsolutePath(file_name) :
    """Compute the absolute path of the file if present in the 'resources' directory"""
    import os
    basepath = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", file_name))


if __name__ == "__main__" :

    text = ""
    with open(getAbsolutePath("test.txt"), 'r') as file :
        for line in file :
            text += line

    restorator = Restorator()

    print("BEFORE = " + text)
    text = restorator.restore(text, RecaserMethod.DNN_CHAR)
    print("AFTER = " + text)
