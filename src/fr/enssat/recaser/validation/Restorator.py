from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser


class Restorator(object) :
    def restore(self, text) :
        text = text.lower()
        recaser = CharDNNRecaser()

        recaser.learn()

        results = recaser.predict(text)

        print(results)

        text_result = ""
        current_index = 0
        for letter in text :
            if results[current_index] == 1 :
                text_result = text_result + letter.upper()
            else :
                text_result = text_result + letter
            current_index += 1

        return text_result
