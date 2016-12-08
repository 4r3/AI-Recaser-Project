from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.Recaser import Recaser

if __name__ == "__main__" :
    #Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_CHAR,"corpus_1/corpus")
   # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD,"corpus_1/corpus")
    Recaser.recase("i love unicorns so much! and you, do you want to have sex with unicorns? oh, yes Bob!", RecaserMethod.CRF_WORD,"corpus_1/corpus")
    #Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD,"corpus_1/corpus")
   # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_CHAR,"corpus_1/corpus")
   # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_WORD,"corpus_1/corpus")