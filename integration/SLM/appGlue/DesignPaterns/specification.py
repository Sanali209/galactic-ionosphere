from abc import ABC, abstractmethod
from typing import Dict


class Specification(ABC):
    @abstractmethod
    def is_satisfied_by(self, candidate):
        pass

    def and_specification(self, specification):
        return AndSpecification(self, specification)

    def or_specification(self, specification):
        return OrSpecification(self, specification)


class AllSpecification(Specification):
    def is_satisfied_by(self, item):
        return True


class AndSpecification(Specification):
    def __init__(self, spec1, spec2):
        self.spec1 = spec1
        self.spec2 = spec2

    def is_satisfied_by(self, candidate):
        return self.spec1.is_satisfied_by(candidate) and self.spec2.is_satisfied_by(candidate)


class OrSpecification(Specification):
    def __init__(self, spec1, spec2):
        self.spec1 = spec1
        self.spec2 = spec2

    def is_satisfied_by(self, candidate):
        return self.spec1.is_satisfied_by(candidate) or self.spec2.is_satisfied_by(candidate)


class NotSpecification(Specification):
    def __init__(self, spec1):
        self.spec1 = spec1

    def is_satisfied_by(self, candidate):
        return not self.spec1.is_satisfied_by(candidate)


class NamedSpecificationSubParser:
    def __init__(self, spec_name, SpecInstance):
        self.spec_name = spec_name
        self.SpecInstance = SpecInstance

    def parse(self, query):
        if query.startswith(self.spec_name):
            return self.SpecInstance(query[len(self.spec_name):])
        else:
            return None


class SpecificationQueryParser:
    def __init__(self):
        self.named_spec_sub_parsers = []

    def parse(self, query):
        return self.parse_query(query)

    def parse_query(self, query):
        if query.startswith("(") and query.endswith(")"):
            return self.parse_query(query[1:-1])
        elif query.startswith("NOT "):
            return NotSpecification(self.parse_query(query[4:]))
        elif query.startswith("AND "):
            return self.parse_and_or_query(query[4:], AndSpecification)
        elif query.startswith("OR "):
            return self.parse_and_or_query(query[3:], OrSpecification)
        else:
            for sub_parser in self.named_spec_sub_parsers:
                spec = sub_parser.parse(query)
                if spec:
                    return spec
            raise Exception("Invalid query: " + query)

    def parse_and_or_query(self, query, spec_type):
        if query.startswith("(") and query.endswith(")"):
            return self.parse_and_or_query(query[1:-1], spec_type)
        elif query.startswith("NOT "):
            return NotSpecification(self.parse_and_or_query(query[4:], spec_type))
        elif query.startswith("AND "):
            return self.parse_and_or_query(query[4:], AndSpecification)
        elif query.startswith("OR "):
            return self.parse_and_or_query(query[3:], OrSpecification)
        else:
            for sub_parser in self.named_spec_sub_parsers:
                spec = sub_parser.parse(query)
                if spec:
                    return spec
            raise Exception("Invalid query: " + query)


class SpecificationBuilder:

    def __init__(self):
        self.named_spec: Dict[str, type] = {}
        self.current_specification = None
        self.combination_specification = None
        self.is_combination_step = False

    def add_specification(self, name, value):
        if self.is_combination_step:
            self.combination_specification.spec2 = self.named_spec.get(name)(value)
            self.current_specification = self.combination_specification
            self.is_combination_step = False
            self.combination_specification = None
            return self
        spec = self.named_spec.get(name)
        self.current_specification = spec(value)
        return self

    def add_and(self):
        self.is_combination_step = True
        self.combination_specification = AndSpecification(self.current_specification, None)
        return self

    def add_or(self):
        self.is_combination_step = True
        self.combination_specification = OrSpecification(self.current_specification, None)
        return self

    def add_not(self):
        self.current_specification = NotSpecification(self.current_specification)
        return self

    def build(self):
        return self.current_specification
