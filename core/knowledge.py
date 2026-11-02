import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


class KnowledgeBase:
    def __init__(self, path="data/knowledge.json"):
        self.path = Path(path)
        self.entries = []
        self.load()

    @staticmethod
    def normalize(text):
        text = unicodedata.normalize("NFKD", str(text).lower())
        text = "".join(
            char for char in text
            if not unicodedata.combining(char)
        )

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = " ".join(text.split())

        corrections = {
            "oque": "o que",
            "pq": "por que",
            "vc": "voce",
            "tb": "tambem",
            "q": "que"
        }

        words = [corrections.get(word, word) for word in text.split()]
        return " ".join(words)

    def load(self):
        if not self.path.exists():
            self.entries = []
            return

        with self.path.open("r", encoding="utf-8") as file:
            self.entries = json.load(file)

    def answer(self, question):
        normalized = self.normalize(question)

        if not normalized:
            return None

        best_answer = None
        best_score = 0.0

        for entry in self.entries:
            for alias in entry.get("aliases", []):
                candidate = self.normalize(alias)

                if normalized == candidate:
                    return entry.get("answer")

                score = SequenceMatcher(
                    None, normalized, candidate
                ).ratio()

                query_words = set(normalized.split())
                alias_words = set(candidate.split())

                if query_words and alias_words:
                    overlap = len(query_words & alias_words) / len(alias_words)
                    score = max(score, overlap * 0.92)

                if score > best_score:
                    best_score = score
                    best_answer = entry.get("answer")

        if best_score >= 0.84:
            return best_answer

        return None
