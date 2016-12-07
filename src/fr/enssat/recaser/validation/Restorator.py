
from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod


class Restorator(object) :

    def restore(self, text_query, method) :
        text_query = text_query.lower() # Insure it's full lower case

        if method == RecaserMethod.DNN_CHAR:
            recaser = CharDNNRecaser()
            recaser.learn()

            results = recaser.predict(text_query)
            text_result = ""
            current_index = 0
            for letter in text_query :
                if results[current_index] == 1 :
                    text_result = text_result + letter.upper()
                else :
                    text_result = text_result + letter
                current_index += 1

            return text_result

        # TODO: Handle other methods
        else:
            raise Exception("Invalid recaser method")




