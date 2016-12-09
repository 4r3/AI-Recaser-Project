from src.fr.enssat.recaser.CRF.CRFRecaser import CRFRecaser
from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser
from src.fr.enssat.recaser.DNN.NamedEntityDNNRecaser import NamedEntityDNNRecaser
from src.fr.enssat.recaser.DNN.WordDNNRecaser import WordDNNRecaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.utils.TextLoader import TextLoader


class Restorator(object) :
    # ==============
    # PUBLIC METHODS
    # ==============

    def restore(self, text_query, method, training_corpus) :
        text_query = text_query.lower()  # Insure it's full lower case

        if method == RecaserMethod.DNN_CHAR :
            recaser = CharDNNRecaser()
            parser = Parser(Parser.MODE_CHARACTER)
            text = TextLoader.get_text(training_corpus)
            elements_learn = parser.read(text, False)
            recaser.learn(elements_learn)
            elements_predict = parser.read(text_query, False)
            results = recaser.predict(elements_predict)
            return self.__restore_chars(elements_predict, results)

        elif method == RecaserMethod.DNN_WORD :
            parser = Parser(Parser.MODE_WORD)
            text = TextLoader.get_text(training_corpus)
            elements_learn = parser.read(text, False)
            recaser = WordDNNRecaser()
            recaser.learn(elements_learn)
            elements_predict = parser.read(text_query, False)
            results = recaser.predict(elements_predict)
            return self.__restore_words(elements_predict, results)

        elif method == RecaserMethod.DNN_NAEN :
            parser = Parser(Parser.MODE_WORD)
            text = TextLoader.get_text(training_corpus)
            elements_learn = parser.read(text, False)
            recaser = NamedEntityDNNRecaser()
            recaser.learn(elements_learn)
            elements_predict = parser.read(text_query, False)
            results = recaser.predict(elements_predict)
            return self.__restore_words(elements_predict, results)

        elif method == RecaserMethod.CRF_CHAR :
            parser = Parser(Parser.MODE_CHARACTER)
            text = TextLoader.get_text(training_corpus)
            elements_learn = parser.read(text, False)
            recaser = CRFRecaser()
            recaser.initModel(elements_learn)
            elements_predict = parser.read(text_query, False)
            results = recaser.predict(elements_predict)
            return self.__restore_chars(elements_predict, results)

        elif method == RecaserMethod.CRF_WORD :
            parser = Parser(Parser.MODE_WORD)
            text = TextLoader.get_text(training_corpus)
            elements_learn = parser.read(text, False)
            recaser = CRFRecaser()
            recaser.initModel(elements_learn)
            elements_predict = parser.read(text_query, False)
            results = recaser.predict(elements_predict)
            return self.__restore_words(elements_predict, results)

    # ===============
    # PRIVATE METHODS
    # ===============

    def __restore_words(self, elements, results) :
        text_result = ""
        current_index = 0
        for element in elements :
            if results[current_index] == RecaserOperation.START_UPPER :
                new = list(element.value)
                new[0] = new[0].upper()
                word = "".join(new)
                text_result += word
            elif results[current_index] == RecaserOperation.FULL_UPPER :
                text_result = text_result + element.value.upper()
            else :
                text_result = text_result + element.value
            current_index += 1

        return text_result

    def __restore_chars(self, elements, results) :
        text_result = ""
        current_index = 0
        for element in elements :
            if results[current_index] == RecaserOperation.START_UPPER :
                text_result = text_result + element.value.upper()
            else :
                text_result = text_result + element.value
            current_index += 1
        return text_result
