from src.fr.enssat.recaser.Recaser import Recaser
from src.fr.enssat.recaser.RecaserMethod import RecaserMethod

if __name__ == "__main__" :









    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_CHAR,"corpus_1/corpus")
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD,"corpus_1/corpus")
    Recaser.recase("I love unicorns so much! And you, do you want to have sex with unicorns? Oh, yes James!", RecaserMethod.CRF_CHAR, ["corpus_11/corpus", "corpus_1/corpus", "corpus_2/corpus"])
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD,"corpus_1/corpus")
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_CHAR,"corpus_1/corpus")
    # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_WORD,"corpus_1/corpus")
