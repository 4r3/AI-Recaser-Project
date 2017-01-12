from src.fr.enssat.recaser.utils.TextLoader import TextLoader
from src.fr.enssat.recaser.Recaser import Recaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod

if __name__ == "__main__" :

    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_CHAR,"corpus_1/corpus")
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD,"corpus_1/corpus")
    # Recaser.recase("You love unicorns so much!", RecaserMethod.CRF_CHAR, ["corpus_11/corpus", "corpus_12/corpus"])
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD,"corpus_1/corpus")
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_CHAR,"corpus_1/corpus")
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_WORD,"corpus_1/corpus")

    #Recaser.learn(RecaserMethod.CRF_WORD, ["corpus_11/corpus", "corpus_12/corpus"], "CRFTest")
    #print(Recaser.predict(RecaserMethod.CRF_WORD, "I love unicorns so much!", "CRFTest"))

    #print(Recaser.learn_predict(RecaserMethod.CRF_WORD, ["corpus_11/corpus", "corpus_12/corpus"], "I love unicorns so much!", "CRFTest2"))

    Recaser.evaluate(RecaserMethod.CRF_WORD, ["corpus_11/corpus", "corpus_12/corpus"], TextLoader.get_text('corpus_13/corpus'))
