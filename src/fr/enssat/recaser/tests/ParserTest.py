from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.utils.TextLoader import TextLoader

if __name__ == "__main__" :
    parser = Parser(Parser.CHARACTER)
    loader = TextLoader()

    text = loader.getText("test.txt",False)

    elements = parser.read(text, True)

    for element in elements :
        print(element)
