import re
import pymongo


class MongoDBQueryParser:
    """
    Main class for parsing MongoDB queries.
    """

    def __init__(self):
        self.operator_parsers = {
            "=": EqualityParser(),
            "!=": InequalityParser(),
            "<": LessThanParser(),
            ">": GreaterThanParser(),
            "<=": LessThanOrEqualParser(),
            ">=": GreaterThanOrEqualParser(),
            "in": InParser(),
            "nin": NotInParser(),
            "$regex": RegexParser(),
            "$text": TextParser(),
            "$exist":ExistParser()
        }

    def parse_query(self, query_string: str):
        """
        Parses a string query into a MongoDB query dictionary.

        Args:
            query_string: The string query to parse.

        Returns:
            A dictionary representing the MongoDB query.
        """
        query_string = query_string.strip(" ")
        if query_string == "":
            return {}
        regex = r"(\(|\)|,|\.|\".*?\"|\S+)"
        tokens = re.findall(regex, query_string)
        if not self.operator_parsers_exist(tokens):
            #query_string=query_string.replace('"', '\"')
            query = {"$text": {"$search": f"{query_string}"}}
            return query
            query_string = f'all $text "{query_string}"'
            tokens = re.findall(regex, query_string)


        query_dict = self._parse_tokens(tokens)
        return query_dict

    def operator_parsers_exist(self, tokens):
        for token in tokens:
            if token in self.operator_parsers:
                return True
        return False

    def _parse_tokens(self, tokens):
        query_stack = []
        current_level = []
        operators = {",": "$and", "or": "$or"}

        for i, token in enumerate(tokens):
            if token == "(":
                query_stack.append(current_level)
                current_level = []
            elif token == ")":
                if not query_stack:
                    raise ValueError("Unmatched parenthesis")
                sub_query = self._parse_tokens(current_level)
                current_level = query_stack.pop()
                current_level.append(sub_query)
            elif token in operators:
                if current_level:
                    prev_expr = current_level.pop()
                    current_level.append({operators[token]: [prev_expr]})
                else:
                    current_level.append(operators[token])
            elif token in self.operator_parsers:
                operator = self.operator_parsers[token]
                field:str = current_level.pop()
                field = field.strip("'")
                field = field.strip("\"")


                value = self._parse_value(tokens[i + 1])  # Get value from next token
                if type(operator) is TextParser:
                    field = "$text"
                current_level.append({field: operator.parse({}, field, value)[field]})
                # Ignore tokens that have been processed (values after operators)
            elif tokens[i - 1] in self.operator_parsers:
                continue
            else:
                current_level.append(token)

        if query_stack:
            raise ValueError("Unmatched parenthesis")
        if len(current_level) == 1:
            return current_level[0]
        else:
            return {"$and": current_level}

    def _parse_value(self, value):
        """
        Helper function to parse a value into the correct type.
        """
        if value.startswith("\'") and value.endswith("\'"):
            value = value[1:-1]
        if value.startswith("\"") and value.endswith("\""):
            value = value[1:-1]
        if value.isdigit():
            return int(value)
        if value.replace(".", "", 1).isdigit():
            return float(value)
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        else:
            return value


# Specialized Parser Classes

class BaseParser:
    """
    Base class for operator parsers.
    """

    def parse(self, query_dict, field, value):
        raise NotImplementedError()


class TextParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict["$text"] = {"$search": value}
        return query_dict


class EqualityParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = value
        return query_dict


class InequalityParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$ne": value}
        return query_dict


class LessThanParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$lt": value}
        return query_dict


class GreaterThanParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$gt": value}
        return query_dict


class LessThanOrEqualParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$lte": value}
        return query_dict


class GreaterThanOrEqualParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$gte": value}
        return query_dict


class InParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$in": value}
        return query_dict


class NotInParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$nin": value}
        return query_dict


class RegexParser(BaseParser):
    def parse(self, query_dict, field, value):
        query_dict[field] = {"$regex": value}
        return query_dict

class ExistParser(BaseParser):
    def parse(self, query_dict, field, value):
        #todo test this
        query_dict[field] = {"$exists": value}
        return query_dict



