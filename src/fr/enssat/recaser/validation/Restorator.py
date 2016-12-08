from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.parser.RecaserOperation import RecaserOperation


class Restorator(object) :
    def restore(self, text_query, method) :
        text_query = text_query.lower()  # Insure it's full lower case

        if method == RecaserMethod.DNN_CHAR :
            recaser = CharDNNRecaser()
            recaser.learn("corpus_1")

            results = recaser.predict(text_query)
            text_result = ""
            current_index = 0
            for letter in text_query :
                if results[current_index] == RecaserOperation.START_UPPER :
                    text_result = text_result + letter.upper()
                else :
                    text_result = text_result + letter
                current_index += 1
            return text_result

        elif method == RecaserMethod.DNN_WORD:
            #TODO
            pass
        elif method == RecaserMethod.CRF_CHAR:
            #TODO
            pass
        elif method == RecaserMethod.CRF_WORD:
            #TODO
            pass
