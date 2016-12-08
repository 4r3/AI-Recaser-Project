from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.utils.TextLoader import TextLoader

if __name__ == "__main__" :
    parser = Parser(Parser.MODE_WORD)
    loader = TextLoader()

    #text = loader.get_text("test.txt", False)
    text = loader.get_text("corpus_776/corpus",False)

    elements = parser.read(text, True)

    # for element in elements :
    #     print(element)
