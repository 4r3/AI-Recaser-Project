from src.fr.enssat.recaser.RecaserMethod import RecaserMethod
from src.fr.enssat.recaser.Recaser import Recaser

if __name__ == "__main__" :
    #Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_CHAR)
    Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.DNN_WORD)
   # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_CHAR)
   # Recaser.recase("i love unicorns! and you, do you like unicorns? yes!", RecaserMethod.CRF_WORD)