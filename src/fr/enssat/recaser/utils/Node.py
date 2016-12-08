class Node :
    def __init__(self, word = None, word_id = None) :
        self.word = word
        self.word_id = word_id
        self.nodes = {}

    def get_id(self, token) :
        char = token[0]
        if (len(token) == 1) :
            if char not in self.nodes :
                self.nodes[char] = Node()

            return self.nodes[char]
        else :
            new_token = token[1 :]
            if char not in self.nodes :
                self.nodes[char] = Node()
            node = self.nodes[char].get_id(token = new_token)

            return node
