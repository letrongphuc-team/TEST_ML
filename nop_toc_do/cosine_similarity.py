import math
import re
import unicodedata
from collections import Counter


class CosineSimilarity:
    """Tính độ tương đồng Cosine giữa 2 chuỗi dựa trên character n-gram."""

    @staticmethod
    def normalize(text):
        text = unicodedata.normalize("NFKD", str(text))
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = text.replace("đ", "d").replace("Đ", "D").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def vectorize(text, ngram_range=(2, 3)):
        s = " %s " % CosineSimilarity.normalize(text)
        vec = Counter()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(s) - n + 1):
                vec[s[i:i + n]] += 1
        return vec

    @staticmethod
    def cosine(vecA, vecB):
        """cos(A,B) = (A.B) / (||A|| * ||B||)"""
        if not vecA or not vecB:
            return 0.0
        dot = sum(v * vecB.get(k, 0) for k, v in vecA.items())
        normA = math.sqrt(sum(v * v for v in vecA.values()))
        normB = math.sqrt(sum(v * v for v in vecB.values()))
        return dot / (normA * normB) if normA and normB else 0.0

    @staticmethod
    def similarity(textA, textB):
        return CosineSimilarity.cosine(
            CosineSimilarity.vectorize(textA),
            CosineSimilarity.vectorize(textB))
