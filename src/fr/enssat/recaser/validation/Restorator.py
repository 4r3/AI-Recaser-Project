from src.fr.enssat.recaser.CRF.CRFRecaser import CRFRecaser
from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.parser.Parser import Parser
from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation
from src.fr.enssat.recaser.utils.TextLoader import TextLoader


class Restorator(object) :
    def restore(self, text_query, method) :
        text_query = text_query.lower()  # Insure it's full lower case

        if method == RecaserMethod.DNN_CHAR :
            recaser = CharDNNRecaser()

            parser = Parser(Parser.MODE_CHARACTER)
            text = TextLoader.get_text("corpus_1/corpus")
            elements_learn = parser.read(text,False)

            recaser.learn(elements_learn)


            elements_predict = parser.read(text_query,False)

            results = recaser.predict(elements_predict)
            return self.__restore_chars(text_query, results)

        elif method == RecaserMethod.DNN_WORD:
            #TODO: Antoine salle feignasse !
            pass
          #  return self.__restore_words()
        elif method == RecaserMethod.CRF_CHAR:
            parser = Parser(Parser.MODE_CHARACTER)
            text = TextLoader.get_text("corpus_1/corpus")
            elements_learn = parser.read(text, False)

            recaser = CRFRecaser()
            recaser.initModel(elements_learn)


            elements_predict = parser.read(text_query,False)

            results = recaser.predict(elements_predict)
            print(results)

            return self.__restore_chars(text_query, results)

        elif method == RecaserMethod.CRF_WORD:
            parser = Parser(Parser.MODE_WORD)
            text = TextLoader.get_text("corpus_1/corpus")
            elements_learn = parser.read(text, False)

            recaser = CRFRecaser()
            recaser.initModel(elements_learn)

            elements_predict = parser.read(text_query, False)

            results = recaser.predict(elements_predict)

            return self.__restore_chars(text_query, results)



    def __restore_words(self, query_lower, results):
        text_result = ""
        current_index = 0
        for word in query_lower.split() :
            if results[current_index] == RecaserOperation.START_UPPER :
                word[0] = word[0].upper()
                text_result = text_result + word
            elif results[current_index] == RecaserOperation.FULL_UPPER :
                text_result = text_result + word.upper()
            else :
                text_result = text_result + word
            current_index += 1
        return text_result

    def __restore_chars(self, query_lower, results):
        text_result = ""
        current_index = 0
        for letter in query_lower:
            if results[current_index] == RecaserOperation.START_UPPER :
                text_result = text_result + letter.upper()
            else :
                text_result = text_result + letter
            current_index += 1
        return text_result