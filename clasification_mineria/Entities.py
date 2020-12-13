import random
from typing import List, Optional, Dict, Union, Tuple
from collections import OrderedDict

from sklearn.preprocessing import label_binarize


class Token:
    def __init__(self, start: int, end: int, text: str, ws: str):
        self._start = start
        self._end = end
        self._text = text
        self._ws = ws

    @property
    def end(self) -> int: return self._end

    @property
    def text(self) -> str: return self._text

    @property
    def start(self) -> int: return self._start

    def __repr__(self) -> str: return self._text

    def __str__(self) -> str: return self._text

    @property
    def ws(self) -> str: return self._text + self._ws

    @property
    def has_ws(self) -> bool: return self._ws != ""

    @ws.setter
    def ws(self, value: str) -> None: self._ws = value


class SpanTokens:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens

    @property
    def to_list(self) -> List[str]:
        return [token.text for token in self._tokens]


class EntityType:
    def __init__(self, ent_type: str):
        self._type = ent_type


class Entity:
    def __init__(self, ent_id: int, ent_did: str, ent_type: EntityType,
                 ent_tokens: List[Token], ent_text: str):
        self._id = ent_id
        self._did = ent_did
        self._type = ent_type
        self._tokens = ent_tokens
        self._text = ent_text

    @property
    def tokens(self) -> List[Token]: return self._tokens

    @property
    def doc_id(self) -> str: return self._did

    @property
    def text(self) -> str: return self._text


class RelationType:
    def __init__(self, rel_type: str):
        self._type = rel_type

    @property
    def value(self) -> str: return self._type


class Relation:
    def __init__(self, rel_id: int, doc_id: str, rel_type: RelationType,
                 entities: OrderedDict):
        self._rel_id = rel_id
        self._doc_id = doc_id
        self._type = rel_type
        self._entities = entities
        _iter = entities.values().__iter__()
        self._head = next(_iter)
        self._tail = next(_iter)

    @property
    def head(self) -> Entity: return self._head

    @property
    def doc_id(self) -> str: return self._doc_id

    @property
    def tail(self) -> Entity: return self._tail

    @property
    def value(self) -> str: return self._type.value


class Document:
    def __init__(self, file_name: str, doc_id: int, text: str,
                 tokens: List[Token]):
        self._name = file_name
        self._doc_id = doc_id
        self._relations = OrderedDict()
        self._entities = OrderedDict()
        self._text = text
        self._tokens = tokens

    def add_entity(self, entity: Optional[Entity]) -> None:
        if entity is not None:
            self._entities[entity.doc_id] = entity

    @property
    def entities(self) -> OrderedDict:
        return self._entities

    def add_relation(self, relation: Relation) -> None:
        self._relations[(relation.head.doc_id,
                         relation.tail.doc_id)] = relation

    @property
    def text(self) -> str:
        return self._text

    @property
    def tokens(self) -> List[Token]:
        return self._tokens

    def get_items(self,
                  negative_count: int = 2) -> List[Tuple[List[str],
                                                         List[str],
                                                         List[str], str]]:
        items = []
        tokens = SpanTokens(self._tokens).to_list
        entities = list(self._entities.values())
        for key in self._relations:
            rel = self._relations[key]
            items.append((tokens, SpanTokens(rel.head.tokens).to_list,
                          SpanTokens(rel.tail.tokens).to_list, rel.value))
            cont = 0
            for i in range(0, negative_count):
                while True and cont < 10:
                    samples = random.sample(entities, 2)
                    cont += 1
                    assert len(samples) == 2, "Error sampling"
                    head = samples[0]
                    tail = samples[1]
                    if (head.doc_id,
                        tail.doc_id) not in self._relations and (
                    tail.doc_id, head.doc_id) not in self._relations:
                        items.append(
                            (tokens, SpanTokens(samples[0].tokens).to_list,
                             SpanTokens(samples[1].tokens).to_list, "None"))
                        break
        return items

    @property
    def file_name(self) -> str:
        return self._name


class Dataset:
    def __init__(self, corpus_name: str):
        self._id = corpus_name
        self._documents = OrderedDict()
        self._relations = OrderedDict()
        self._entities = OrderedDict()
        self._entity_types = OrderedDict()
        self._relation_types = OrderedDict()
        self._did = 0
        self._rid = 0
        self._eid = 0
        self.create_relation_type("None")

    def create_document(self, filename: str, text: str,
                        tokens: List[Token]) -> Document:
        doc = Document(file_name=filename, doc_id=self._did, text=text,
                       tokens=tokens)
        self._documents[self._did] = doc
        self._did += 1
        return doc

    def create_entity(self, ent_did: str, ent_type: EntityType,
                      ent_tokens: List[Token], ent_text: str) -> Entity:
        ent = Entity(ent_id=self._eid, ent_did=ent_did, ent_type=ent_type,
                     ent_tokens=ent_tokens, ent_text=ent_text)
        self._entities[self._eid] = ent
        self._eid += 1
        return ent

    def create_entity_type(self, entity_type: str) -> EntityType:
        if entity_type not in self._entity_types:
            self._entity_types[entity_type] = EntityType(entity_type)
        return self._entity_types[entity_type]

    def create_relation_type(self, rel_type: str) -> RelationType:
        if rel_type not in self._relation_types:
            self._relation_types[rel_type] = RelationType(rel_type)
        return self._relation_types[rel_type]

    def create_relation(self, rel_id: str, rel_type: RelationType,
                        rel_entities: OrderedDict) -> Relation:
        rel = Relation(self._rid, rel_id, rel_type, rel_entities)
        self._relations[self._rid] = rel
        self._rid += 1
        return rel

    def get_data(self) -> List[List[str]]:
        return [SpanTokens(doc.tokens).to_list for doc in
                self._documents.values()]

    def get_raw_data(self,
                     proportion: int,
                     rel_types: List[str]) -> Dict[str, Union[List[str], str]]:
        raw_data = {
            "tokens": [],
            "head": [],
            "tail": [],
            "class": []
        }
        for key in self._documents:
            items = self._documents[key].get_items(proportion)
            for tokens, head, tail, relation in items:
                raw_data["tokens"].append(tokens)
                raw_data["head"].append(head)
                raw_data["tail"].append(tail)
                raw_data["class"].append(relation)
        elements = list(
            zip(raw_data["tokens"], raw_data["head"], raw_data["tail"],
                raw_data["class"]))
        random.shuffle(elements)
        tok, head, tail, value = zip(*elements)
        value = label_binarize(value, rel_types)
        return {
            "tokens": tok,
            "head": head,
            "tail": tail,
            "class": value
        }

    def get_summary(self) -> str:
        return f"""
Num Documents: {len(self._documents)}
Num Relations: {len(self._relations)}
Num Entities: {len(self._entities)}
Num RelationTypes: {len(self._relation_types)}
"""

    def get_relation_types(self) -> List[str]:
        return [self._relation_types[key].value for key in
                self._relation_types]
