import os


class TextLoader(object) :
    # ================
    # PUBLIC FUNCTIONS
    # ================

    def get_text(self, resource, absolute_path = False) :
        if not absolute_path :
            resource = self.__get_absolute_path(resource)
        return self.__concat_lines(resource)

    # =================
    # PRIVATE FUNCTIONS
    # =================

    def __get_absolute_path(self, resource) :
        """Compute the absolute path of the file if present in the 'resources' directory"""
        basepath = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", resource))

    def __concat_lines(self, resource_path) :
        text = ""
        with open(resource_path, 'r') as file :
            for line in file :
                text += line
        return text
