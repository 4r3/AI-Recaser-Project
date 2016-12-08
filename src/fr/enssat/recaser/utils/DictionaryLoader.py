import os
import yaml

from src.fr.enssat.recaser.utils.Dictionary import Dictionary


class DictionaryLoader(object) :
    # ================
    # PUBLIC FUNCTIONS
    # ================

    @staticmethod
    def load_dictionary(resource, absolute_path=False):
        if not absolute_path:
            resource = DictionaryLoader.__get_absolute_path(resource)
        try:
            return DictionaryLoader.__load_dictionary(resource)
        except FileNotFoundError:
            print("Dictionary not found... creating a default one...")
            new_dictionary = Dictionary()
            DictionaryLoader.save_dictionary(new_dictionary, "default_dictionary.yaml")
            return new_dictionary

    @staticmethod
    def save_dictionary(dictionary, resource, absolute_path=False):
        if not absolute_path:
            resource = DictionaryLoader.__get_absolute_path(resource)
        return DictionaryLoader.__save_dictionary(resource,dictionary)

    # =================
    # PRIVATE FUNCTIONS
    # =================

    @staticmethod
    def __get_absolute_path(resource):
        """Compute the absolute path of the file if present in the 'resources' directory"""
        base_path = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(base_path, "..", "..", "..", "..", "..", "resources", resource))

    @staticmethod
    def __load_dictionary(resource_path):
        with open(resource_path, 'r') as file:
            dictionary = yaml.load(file)
        return dictionary

    @staticmethod
    def __save_dictionary(resource_path,dictionary):
        with open(resource_path, 'w') as file:
            yaml.dump(dictionary, file)
