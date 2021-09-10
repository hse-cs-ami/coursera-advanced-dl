import torch


class Alphabet(object):
    blank_token = 0

    def __init__(self, tokens=''):
        self.index_to_token = {self.blank_token: ''}
        self.index_to_token.update({i + self.blank_token + 1: tokens[i]
                                    for i in range(len(tokens))})
        self.token_to_index = {token: index
                               for index, token in self.index_to_token.items()}

    def __len__(self):
        return len(self.index_to_token)

    def __contains__(self, token):
        return token in self.token_to_index

    def string_to_indices(self, string):
        return torch.tensor([self.token_to_index[token] for token in string
                             if token in self.token_to_index], dtype=torch.int32)

    def indices_to_string(self, indices):
        return ''.join(self.index_to_token[index.item()] for index in indices)

    def decode(self, log_prob):
        indices = torch.argmax(log_prob, dim=0)
        indices = torch.unique_consecutive(indices)
        indices = indices[indices != self.blank_token]
        return self.indices_to_string(indices)
