from src.fr.enssat.recaser.DNN.CharDNNRecaser import CharDNNRecaser


class Restorator(object):

    def restore(self, text):
        text = text.lower()
        recaser = CharDNNRecaser()

        results = recaser.learn_and_return()

        current_index = 0
        for letter in text:
            if results[current_index] == 1:
                letter = letter.upper()
            current_index += 1

        return text

