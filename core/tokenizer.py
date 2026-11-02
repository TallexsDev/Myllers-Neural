import json


class Tokenizer:
    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}

    def train(self, text):
        self.chars = ["<UNK>"] + sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, tokens):
        return "".join(
            self.itos.get(int(token), "")
            for token in tokens
            if int(token) != 0
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.chars, file, ensure_ascii=False)

    def load(self, path):
        with open(path, encoding="utf-8") as file:
            self.chars = json.load(file)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)
