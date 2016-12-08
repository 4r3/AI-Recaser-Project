import os


class TextLoader(object) :
    # ================
    # PUBLIC FUNCTIONS
    # ================

    def getText(self, resource, absolute_path = False) :
        if not absolute_path :
            resource = self.__getAbsolutePath(resource)
        return self.__concatLines(resource)

    # =================
    # PRIVATE FUNCTIONS
    # =================

    def __getAbsolutePath(self, resource) :
        """Compute the absolute path of the file if present in the 'resources' directory"""
        basepath = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(basepath, "..", "..", "..", "..", "..", "resources", resource))

    def __concatLines(self, resource_path) :
        text = ""
        with open(resource_path, 'r') as file :
            for line in file :
                text += line
        return text
