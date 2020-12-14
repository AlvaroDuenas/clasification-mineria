from itertools import cycle
from typing import List, Union
from gensim.sklearn_api import D2VTransformer
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def dummy_fun(doc):
    return doc


class ClassifierFactory:
    """
    The classifier factory
    """
    names = ["RandomForest", "SVM"]

    @staticmethod
    def get_classifier(name: str):
        """
        Returns the requested classifier
        Args:
            name: The requested classifier's name

        Returns:
            The Classifier

        Raises:
            ValueError: If not found

        """
        if name == "SVM":
            return SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                 random_state=42,
                                 max_iter=5, tol=None)
        elif name == "RandomForest":
            # return MultinomialNB()
            return RandomForestClassifier()
        else:
            raise ValueError


class TransformerFactory:
    """
    The transformer generator
    """
    models = ["tfidf", "doc2vec"]

    @staticmethod
    def get_transformer(name: str) -> Union[TfidfVectorizer, D2VTransformer]:
        """
        Returns the requested transformer
        Args:
            name: The requested transformer's name

        Returns:
            The requested transformer

        """
        if name == "tfidf":
            return TfidfVectorizer(
                use_idf=True,
                analyzer='word',
                tokenizer=dummy_fun,
                preprocessor=dummy_fun,
                stop_words=None,
                token_pattern=None)
        elif name == "doc2vec":
            return D2VTransformer(dm=1, size=100, window=5, iter=10)
        else:
            raise ValueError


class Transformer(BaseEstimator, TransformerMixin):
    """
    The transformer, converts tokens to vectorized data.

    Attributes:
        _transformer (Union[TfidfVectorizer, D2VTransformer]): The converter
    """

    def __init__(self, name: str = "doc2vec"):
        self._transformer = TransformerFactory.get_transformer(name)

    def fit(self, x, y=None) -> 'Transformer':
        """
        Trains the converter
        Args:
            x: Train data
            y: Test data

        Returns:
            Returns itself
        """
        self._transformer.fit(x)
        return self

    def transform(self, x, y=None):
        """
        Transforms the train data
        Args:
            x: The train data
            y: The test data

        Returns:
            The vectorized data

        """
        tokens = self._transformer.transform(x["tokens"])
        head = self._transformer.transform(x["head"])
        tail = self._transformer.transform(x["tail"])
        if isinstance(self._transformer, TfidfVectorizer):
            return sparse.hstack((tokens,
                                  head,
                                  tail))
        elif isinstance(self._transformer, D2VTransformer):
            return pd.DataFrame(tokens).join(pd.DataFrame(head),
                                             rsuffix="_").join(
                pd.DataFrame(tail), lsuffix="_").values

    def fit_transform(self, x, y=None, **fit_params):
        """
        Trains and transforms together
        Args:
            x: Train data
            y: Test data
            **fit_params: Ignored Args

        Returns:
            The vectorized train data
        """
        return self.fit(x["tokens"]).transform(x)

    def len_vocab(self) -> int:
        """
        Gets the trained transformer's vocabs length
        Returns:
            The length
        """
        if isinstance(self._transformer, TfidfVectorizer):
            return len(self._transformer.vocabulary_)
        elif isinstance(self._transformer, D2VTransformer):
            return len(self._transformer.gensim_model.wv.vocab)


class Classifier:
    """
    The pipeline that transforms the input and performs the classification

    Attributes:
        _transformer (Transformer): The transformer
        _classifier (OneVsRestClassifier): The multiclass classifier
        _pipe (Pipeline): The pipeline
    """

    def __init__(self, transformer: str = "tfidf",
                 classifier: str = "RandomForest"):
        self._transformer = Transformer(transformer)
        self._classifier = OneVsRestClassifier(
            ClassifierFactory.get_classifier(classifier))
        self._pipe = make_pipeline(self._transformer, self._classifier)

    def train(self, x_train, y_train) -> None:
        """
        Trains the pipeline
        Args:
            x_train: Train attributes
            y_train: Train class
        """
        self._pipe.fit(x_train, y_train)

    def len_vocabulary(self) -> int:
        """
        The transformer's vocabs length
        Returns:
            The length
        """
        return self._pipe[0].len_vocab()

    def dev(self, x_test, y_test, rel_types: List[str],
            output_path: str = None) -> pd.DataFrame:
        """
        Tests the trained model
        Args:
            x_test: dev attributes
            y_test: dev class
            rel_types: class values
            output_path: Folder to store generated files

        Returns:
            The info dataframe

        """
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

    def _roc(self, x_test, y_test, rel_types: List[str],
             output_path: str) -> None:
        """
        Performs roc over the data and the model
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        Args:
            x_test: Dev attributes values
            y_test: Dev class values
            rel_types: Class Values
            output_path: Folder to store generated files
        """
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
