from collections import defaultdict
from math import log


class NaiveBayes:
    def __init__(self):
        pass

    @staticmethod
    def fit(data: list, target: list):
        classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for feats, label in zip(data, target):
            classes[label] += 1  # count classes frequencies
            for feat in feats.split():
                freq[label, feat] += 1  # count features frequencies
                # if feat == 'FA':
                #     print(label, )

        for label, feat in freq:  # normalize features frequencies
            freq[label, feat] /= classes[label]

        for c in classes:  # normalize classes frequencies
            classes[c] /= len(target)

        return classes, freq  # return P(C) and P(O|C)

    @staticmethod
    def predict(classifier, feats):
        classes, prob = classifier
        # calculate argmin(-log(C|O))
        return (
            min(
                classes.keys(),
                key=lambda cl: -log(classes[cl]) + sum(
                    -log(prob.get((cl, feat), 10 ** (-7)))
                    for feat in feats.split()
                )
            )
        )
