from src.fr.enssat.recaser.validation.Restorator import Restorator


class Recaser(object) :

    @staticmethod
    def recase(text_query, approach, training_corpus="corpus_1/corpus"):
        """Recase the given text_query using the given approach."""

        tmp_text = text_query
        text_query = text_query.lower()

        restorator = Restorator()
        result = restorator.restore(text_query, approach, training_corpus)

        print("------------------------------ RECASING...")
        print("Original input   = ", tmp_text)
        print("Lower case input = ", text_query)
        print("Predicted output = ", result)

        #TODO: Add measures
