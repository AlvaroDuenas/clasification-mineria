from abc import ABC, abstractmethod
from typing import List, Type
from spacy import load as spacy_load
from spacy.cli import download as spacy_download
from spacy.util import compile_infix_regex, compile_suffix_regex, \
    compile_prefix_regex

from clasification_mineria.Entities import Token


class Tokenizer(ABC):
    """
    The abstract tokenizer

    Attributes:
        _model_name (str): Name of each option
    """
    models = []

    def __init__(self, model_name: str):
        self._model_name = model_name

    @abstractmethod
    def tokenize(self, text: str) -> List[Token]: ...


class SpacyTokenizer(Tokenizer):
    """
    The abstract tokenizer

    Attributes:
        _nlp : The Spacy's natural language processor
    """
    models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._nlp = SpacyTokenizer.load_nlp(model_name=model_name)

        suffixes = self._nlp.Defaults.suffixes + (r"[\.]", r"[+]")
        suffix_regex = compile_suffix_regex(suffixes)
        self._nlp.tokenizer.suffix_search = suffix_regex.search

        infix = self._nlp.Defaults.infixes + tuple(
            [r"(?<=[A-Za-z])[\[\.+\-\*^](?=[0-9-])",
             r"(?<=[0-9])[\[\.+\-\*^](?=[A-Za-z-])",
             r"[\.]", r"(?<=[0-9])(?=[A-Za-z-])", r"(?<=[A-Za-z])(?=[0-9-])"])
        infix_re = compile_infix_regex(infix)
        self._nlp.tokenizer.infix_finditer = infix_re.finditer

        prefixes = self._nlp.Defaults.suffixes + tuple(r"[\.]")
        prefix_regex = compile_prefix_regex(prefixes)
        self._nlp.tokenizer.prefix_search = prefix_regex.search

    @staticmethod
    def load_nlp(model_name: str):
        """
        Loads the spacy's model and downloads it if needed.
        Args:
            model_name: Name of the model

        Returns:
            The generated model
        """
        try:
            return spacy_load(model_name)
        except OSError:
            print('Downloading language model for the spaCy POS tagger\n'
                  "(don't worry, this will only happen once)")
            spacy_download(model_name)
            return spacy_load(model_name)

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenizes the requested text
        Args:
            text: The string to be tokenized

        Returns:
            The generated tokens
        """
        return [Token(token.idx, token.idx + len(token), token.text,
                      token.whitespace_) for token in self._nlp(text)]


class TokenizerFactory:
    """
    The tokenizer generator
    """
    tokenizers = ["spacy"]

    @staticmethod
    def get_tokenizer(name: str) -> Type[Tokenizer]:
        """
        Given a tokenizer name returns the related Tokenizer.
        Args:
            name: Tokenizer name

        Returns:
            The related Tokenizer type

        Raises:
            ValueError: If not found
        """
        if name == "spacy":
            return SpacyTokenizer
        else:
            raise ValueError
