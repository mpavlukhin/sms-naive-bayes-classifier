from collections import defaultdict
from math import log


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.freq = None

    def fit(self, data: list, target: list):
        classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for feats, label in zip(data, target):
            classes[label] += 1  # count classes frequencies
            for feat in feats.split():
                freq[label, feat] += 1  # count features frequencies

        for label, feat in freq:  # normalize features frequencies
            freq[label, feat] /= classes[label]

        for c in classes:  # normalize classes frequencies
            classes[c] /= len(target)

        self.classes = classes # P(C)
        self.freq = freq # P(O|C)

    def predict(self, data: list):
        predictions = []
        for value in data:
            # calculate argmin(-log(C|O))
            predictions.append(
                min(
                    self.classes.keys(),
                    key=(
                        lambda cl:
                        -log(self.classes[cl]) + sum(
                            -log(self.freq.get((cl, feat), 10 ** (-7)))
                            for feat in value.split()
                        )
                    )
                )
            )
        return predictions
