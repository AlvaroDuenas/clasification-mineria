from abc import ABC, abstractmethod
from typing import List, Type
from spacy import load as spacy_load
from spacy.cli import download as spacy_download
from spacy.util import compile_infix_regex, compile_suffix_regex, \
    compile_prefix_regex

from clasification_mineria.Entities import Token


class Tokenizer(ABC):
    models = []

    def __init__(self, model_name: str):
        self._model_name = model_name

    @abstractmethod
    def tokenize(self, text: str) -> List[Token]: ...


class SpacyTokenizer(Tokenizer):
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
        try:
            return spacy_load(model_name)
        except OSError:
            print('Downloading language model for the spaCy POS tagger\n'
                  "(don't worry, this will only happen once)")
            spacy_download(model_name)
            return spacy_load(model_name)

    def tokenize(self, text: str) -> List[Token]:
        return [Token(token.idx, token.idx + len(token), token.text,
                      token.whitespace_) for token in self._nlp(text)]


class TokenizerFactory:
    tokenizers = ["spacy"]

    @staticmethod
    def get_tokenizer(name: str) -> Type[Tokenizer]:
        if name == "spacy":
            return SpacyTokenizer
        else:
            raise ValueError
