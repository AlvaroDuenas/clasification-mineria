from itertools import cycle
from typing import List

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc, roc_curve, f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def dummy_fun(doc):
    return doc


class ClassifierFactory:
    names = ["Bayes", "SVM"]

    @staticmethod
    def get_classifier(name: str):
        if name == "SVM":
            return SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                 random_state=42,
                                 max_iter=5, tol=None)
        elif name == "Bayes":
            return MultinomialNB()
        else:
            raise ValueError


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_idf=True, stop_words=None):
        self._tfidf = TfidfVectorizer(
            use_idf=use_idf,
            analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            stop_words=stop_words,
            token_pattern=None)

    def fit(self, x, y=None):
        self._tfidf.fit(x)
        return self

    def transform(self, x, y=None):
        return sparse.hstack((self._tfidf.transform(x["tokens"]),
                              self._tfidf.transform(x["head"]),
                              self._tfidf.transform(x["tail"])))

    def fit_transform(self, x, y=None, **fit_params):
        return self.fit(x["tokens"]).transform(x)


class Classifier:
    def __init__(self, classifier="Bayes", **kwargs):
        self._transformer = Transformer(**kwargs)
        self._classifier = OneVsRestClassifier(
            ClassifierFactory.get_classifier(classifier))
        self._pipe = make_pipeline(self._transformer, self._classifier)

    def train(self, x_train, y_train):
        self._pipe.fit(x_train, y_train)

    def dev(self, x_test, y_test, rel_types: List[str],
            output_path: str = None):
        if hasattr(self._pipe, "decision_function"):
            self._roc(x_test, y_test, rel_types, output_path)
        y_pred = self._pipe.predict(x_test)
        plt.figure()
        report = classification_report(y_test, y_pred,
                                       target_names=rel_types,
                                       output_dict=True)
        data_frame = pd.DataFrame(report)

        sns.heatmap(data_frame.iloc[:-1, :].T, annot=True).set_title(
            "Classification Report")

        if output_path is not None:
            data_frame.to_csv(output_path + "report.csv")
            plt.savefig(output_path + "report.png")
        else:
            plt.show()
        plt.close()
        return data_frame

    def _roc(self, x_test, y_test, rel_types: List[str], output_path: str):
        y_score = self._pipe.decision_function(x_test)
        n_classes = len(rel_types)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        lw = 2
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
                                                  y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(rel_types[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'ROC curve for Relation Classification With Annotated Entities')
        plt.legend(loc="lower right")
        if output_path is not None:
            plt.savefig(output_path + "roc.png")
        else:
            plt.show()
        plt.close()
