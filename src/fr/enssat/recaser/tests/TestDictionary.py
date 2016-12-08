import yaml

from src.fr.enssat.recaser.utils.Dictionary import Dictionary
from src.fr.enssat.recaser.utils.DictionaryLoader import DictionaryLoader

dict1 = Dictionary()

print(dict1.get_id("a"))
print(dict1.get_id("d"))
print(dict1.get_id("a"))
print(dict1.get_id("c"))
print(dict1.get_id("j"))
print(dict1.get_id("e"))
print(dict1.get_id("f"))
print(dict1.get_id("test"))
print(dict1.get_id("testing"))
print(dict1.get_id("test"))

dump = yaml.dump(dict)

DictionaryLoader.save_dictionary(dict1, "test.yml")

dict2 = DictionaryLoader.load_dictionary("test.yml")

print(dict2.get_id("test"))
