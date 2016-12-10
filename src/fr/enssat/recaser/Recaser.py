from fr.enssat.recaser.DNN.NamedEntityDNNRecaser import NamedEntityDNNRecaser
from fr.enssat.recaser.DNN.WordDNNRecaser import WordDNNRecaser
from src.fr.enssat.recaser.CRF.CRFRecaser import CRFRecaser
from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser
from src.fr.enssat.recaser.validation.Restorator import Restorator
from src.fr.enssat.recaser.validation.Tester import Tester
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.utils.TextLoader import TextLoader
from src.fr.enssat.recaser.validation.Validation import Validation


class Recaser(object) :
    
    @staticmethod
    def recase(text_query, approach, training_corpus = ["corpus_1/corpus"]) :
        """Recase the given text_query using the given approach."""

        original_text = text_query
        text_query = text_query.lower()

        restorator = Restorator()
        result = restorator.restore(text_query, approach, training_corpus)

        print("------------------------------ RECASING...")
        print("Original input   = ", original_text)
        print("Lower case input = ", text_query)
        print("Predicted output = ", result)

    @staticmethod
    def learn(approach, training_corpus=["corpus_1/corpus"], file=None):
        """Le CRF prend un nom de fichier en entrée et est enregistré dans le dossier models"""
        if approach == RecaserMethod.DNN_CHAR :
            return 0

        elif approach == RecaserMethod.DNN_WORD :
            return 0

        elif approach == RecaserMethod.DNN_NAEN :
            return 0

        elif approach == RecaserMethod.CRF_CHAR :
            parser = Parser(Parser.MODE_CHARACTER)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = CRFRecaser()
            if file is not None:
                recaser.setFile(file)
            recaser.initModel(elements_learn)
        elif approach == RecaserMethod.CRF_WORD :
            parser = Parser(Parser.MODE_WORD)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = CRFRecaser()
            if file is not None:
                recaser.setFile(file)
            recaser.initModel(elements_learn)

    @staticmethod
    def predict(approach, text_query="I like reading. What about you?", file=None):
        """Le CRF prend un nom de fichier en entrée et est enregistré dans le dossier models"""
        text_query = text_query.lower()
        if approach == RecaserMethod.DNN_CHAR :
            return 0

        elif approach == RecaserMethod.DNN_WORD :
            return 0

        elif approach == RecaserMethod.DNN_NAEN :
            return 0

        elif approach == RecaserMethod.CRF_CHAR :
            parser = Parser(Parser.MODE_CHARACTER)
            recaser = CRFRecaser()
            elements_predict = parser.read(text_query, False)
            if file is not None:
                recaser.setFile(file)
            return recaser.predict(elements_predict)
        elif approach == RecaserMethod.CRF_WORD :
            parser = Parser(Parser.MODE_WORD)
            recaser = CRFRecaser()
            elements_predict = parser.read(text_query, False)
            if file is not None:
                recaser.setFile(file)
            return recaser.predict(elements_predict)

    @staticmethod
    def learn_predict(approach, training_corpus=["corpus_1/corpus"],
                      text_query="I like reading. What about you?", file=None):
        """Le CRF prend un nom de fichier en entrée et est enregistré dans le dossier models"""
        Recaser.learn(approach, training_corpus, file)
        return Recaser.predict(approach, text_query, file)

    @staticmethod
    def evaluate(approach, training_corpus=["corpus_1/corpus"],
                      text_query="I like reading. What about you?", file=None):
        """Le CRF prend un nom de fichier en entrée et est enregistré dans le dossier models"""
        if approach == RecaserMethod.DNN_CHAR :
            parser = Parser(Parser.MODE_CHARACTER)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = CharDNNRecaser(name=file)
            recaser.learn(elements_learn,epochs=20)
            elements_predict = parser.read(text_query, False)
            correct = [str(message.operation) for message in elements_predict]
            predict = [str(message) for message in recaser.predict(elements_predict)]
            tester = Tester()
            tester.test(correct, predict, Validation.CHAR)
        elif approach == RecaserMethod.DNN_WORD :
            parser = Parser(Parser.MODE_WORD)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = WordDNNRecaser(name=file)
            recaser.learn(elements_learn, epochs=20)
            elements_predict = parser.read(text_query, False)
            correct = [str(message.operation) for message in elements_predict]
            predict = [str(message) for message in recaser.predict(elements_predict)]
            tester = Tester()
            tester.test(correct, predict, Validation.WORD)

        elif approach == RecaserMethod.DNN_NAEN :
            parser = Parser(Parser.MODE_WORD)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = NamedEntityDNNRecaser(name=file)
            recaser.learn(elements_learn,epochs=100)
            elements_predict = parser.read(text_query, False)
            correct = [str(message.operation) for message in elements_predict]
            predict = [str(message) for message in recaser.predict(elements_predict)]
            tester = Tester()
            tester.test(correct, predict, Validation.WORD)

        elif approach == RecaserMethod.CRF_CHAR :
            parser = Parser(Parser.MODE_CHARACTER)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = CRFRecaser()
            if file is not None:
                recaser.setFile(file)
            recaser.initModel(elements_learn)
            elements_predict = parser.read(text_query, False)
            [correct, predict] = recaser.predictAndTest(elements_predict)
            tester = Tester()
            tester.test(correct, predict, Validation.CHAR)
        elif approach == RecaserMethod.CRF_WORD :
            parser = Parser(Parser.MODE_WORD)
            elements_learn = []
            for training in training_corpus:
                text = TextLoader.get_text(training)
                elements_learn.extend(parser.read(text, False))
            recaser = CRFRecaser()
            if file is not None:
                recaser.setFile(file)
            recaser.initModel(elements_learn)
            elements_predict = parser.read(text_query, False)
            [correct, predict] = recaser.predictAndTest(elements_predict)
            tester = Tester()
            tester.test(correct, predict, Validation.WORD)