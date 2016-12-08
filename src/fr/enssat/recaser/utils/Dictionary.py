from src.fr.enssat.recaser.utils.Node import Node


class Dictionary(object) :
    def __init__(self) :
        self.size = 0
        self.nodes = {}

    def get_id(self, token) :
        char = token[0]
        node = None
        if (len(token) == 1) :
            if char not in self.nodes :
                self.nodes[char] = Node(word = token, word_id = self.size)
                self.size += 1

            node = self.nodes[char]
        else :
            new_token = token[1 :]
            if char not in self.nodes :
                self.nodes[char] = Node()
            node = self.nodes[char].get_id(token = new_token)

        if node.word_id == None :
            node.word_id = self.size
            node.word = token
            self.size += 1

        return node.word_id

