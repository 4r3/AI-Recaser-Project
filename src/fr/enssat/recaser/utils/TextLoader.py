import os


class TextLoader(object):
    # ================
    # PUBLIC FUNCTIONS
    # ================

    @staticmethod
    def get_text(resource, absolute_path = False) :
        if not absolute_path :
            resource = TextLoader.__get_absolute_path(resource)
        return TextLoader.__concat_lines(resource)

    # =================
    # PRIVATE FUNCTIONS
    # =================

    @staticmethod
    def __get_absolute_path(resource):
        """Compute the absolute path of the file if present in the 'resources' directory"""
        basepath = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", resource))

    @staticmethod
    def __concat_lines(resource_path) :
        text = ""
        with open(resource_path, 'r') as file :
            for line in file :
                text += line
        return text
