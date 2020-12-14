import codecs
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Type

from clasification_mineria.Entities import Dataset, Entity, Relation, Document
from clasification_mineria.Tokenizer import Tokenizer


class Reader(ABC):
    """
    Format abstract reader.
    """
    @staticmethod
    @abstractmethod
    def load(file_path: str, tokenizer: Tokenizer) -> Dataset: ...


class Standoff(Reader):
    """
    Standoff format reader
    """
    annotation_extensions = ['ann', 'a1', 'a2']
    ignored_entities = ['Title', 'Paragraph']
    split_paragraphs = False

    @staticmethod
    def load(file_path: str, tokenizer: Tokenizer) -> Dataset:
        """
        Loads the folder dataset with the given tokenizer if needed
        Args:
            file_path: Folder path
            tokenizer: Tokenizer for splitting text

        Returns:
            The generated dataset
        """
        corpus_name = os.path.basename(os.path.dirname(file_path))
        corpus = Dataset(corpus_name)
        if os.path.isdir(file_path):
            directory = file_path
            filenames = sorted(list(os.listdir(directory)))
            for filename in filenames:
                if filename.endswith(".txt"):
                    abs_path = os.path.join(directory, filename)
                    # print(filename)
                    Standoff.load_document(abs_path, filename, corpus,
                                           tokenizer)
        else:
            Standoff.load_document(file_path, corpus_name, corpus, tokenizer)
        return corpus

    @staticmethod
    def get_annotation_files(file_path: str) -> List[str]:
        """
        Gets the annotation files of the txt file
        Args:
            file_path: file path

        Returns:
            the annotation files paths

        Raises:
            AssertionError: If an error occurs

        """
        assert file_path.endswith(".txt")
        base = file_path[:-4]
        annotation_files = ["%s.%s" % (base, ext) for ext in
                            Standoff.annotation_extensions]
        return [filename for filename in annotation_files if
                os.path.isfile(filename)]

    @staticmethod
    def load_entity(line: str,
                    doc: Document, dataset: Dataset) -> Optional[Entity]:
        """
        Loads a entity from the given line into the dataset
        Args:
            line: The Annotation Entity
            doc: The Corresponding Document Object
            dataset: The Corresponding Dataset

        Returns:
            The generated Entity if it's not ignored

        Raises:
            StopIteration: If there arent more tokens in the doc.
            AssertionError: If an error occurs
        """
        split = line.split('\t')
        i = str(split[1]).index(' ')
        ent_id = split[0]
        ent_text = split[2]
        ent_type_name = split[1][:i]
        if ent_type_name in Standoff.ignored_entities:
            return None
        ent_type = dataset.create_entity_type(ent_type_name)

        ent_tokens = []
        try:
            _iter = doc.tokens.__iter__()
            token = next(_iter)
            for start, end in map(lambda c: tuple(map(int, c.split(' '))),
                                  split[1][i + 1:].split(";")):
                while token.end <= end:
                    if start <= token.start:
                        ent_tokens.append(token)
                    token = next(_iter)
                try:
                    ent_tokens[-1].ws = " "
                except IndexError:
                    print(doc.file_name)
                    print(line)
                    print(doc.tokens)
        except StopIteration:
            print("Iteration Stopped")
        ent_tokens[-1].ws = ""
        check_text = "".join([token.ws for token in ent_tokens]).strip()
        if len(check_text) != len(ent_text):
            for token in ent_tokens:
                if str(token) + " " in ent_text and not token.has_ws:
                    token.ws = " "
            check_text = "".join([token.ws for token in ent_tokens]).strip()
            error_msg = f"""
No coinciden las entidades en {doc.file_name} 
{line}
{ent_text}
{check_text}
{ent_tokens}
{str(doc.tokens)}
{doc.text}"""
            assert ent_text == check_text, error_msg
        return dataset.create_entity(ent_id, ent_type, ent_tokens,
                                     ent_text)

    @staticmethod
    def load_relation(line: str,
                      doc: Document, dataset: Dataset) -> Relation:
        assert line[0] == 'E' or line[
            0] == 'R', f"""
ERROR in {doc.file_name}. 
In line: {line}
Relation input should start with a E or R"""
        split = line.split("\t")
        rel_id = split[0]
        relation = split[1].split(" ")
        rel_type = dataset.create_relation_type(relation[0])
        rel_entities = OrderedDict()
        for name, entity_id in [tuple(entity.split(":")) for entity in
                                relation[1:]]:
            assert entity_id.startswith("T"), f"""
ERROR in {doc.file_name}. 
In line: {line}
Complex relations are not allowed"""
            rel_entities[entity_id] = doc.entities[entity_id]
        assert len(rel_entities.values()) == 2, f"""
ERROR in {doc.file_name}. 
In line: {line}
Complex relations are not allowed"""
        return dataset.create_relation(rel_id, rel_type, rel_entities)

    @staticmethod
    def load_document(file_path: str, filename: str, dataset: Dataset,
                      tokenizer: Tokenizer) -> None:
        """
        Loads the current document into the dataset
        Args:
            file_path: The current document file path
            filename: The name of the file
            dataset: The dataset object
            tokenizer: The text tokenizer if needed
        """
        with codecs.open(file_path, "r", "utf-8") as f:
            text = f.read()
            text = re.sub('\n', ' ', text)
        doc = dataset.create_document(filename, text, tokenizer.tokenize(text))
        for annotation_file in Standoff.get_annotation_files(file_path):
            with codecs.open(annotation_file, "r", "utf-8") as f:
                for line in f:
                    try:
                        line = str(line.strip())
                        if line.startswith('T'):
                            doc.add_entity(
                                Standoff.load_entity(line, doc, dataset))
                        elif line.startswith('E') or line.startswith('R'):
                            doc.add_relation(
                                Standoff.load_relation(line, doc, dataset))
                    except AssertionError:
                        pass
                    except KeyError:
                        pass


class ReaderFactory:
    """
    Generates the requested reader
    """
    readers = ["standoff"]

    @staticmethod
    def get_reader(name: str) -> Type[Reader]:
        """
        Generates the requested reader

        Args:
            name: the reader's name

        Returns:
            The Requested Reader's Type

        Raises:
            ValueError: If not found
        """

        if name == "standoff":
            return Standoff
        else:
            raise ValueError
