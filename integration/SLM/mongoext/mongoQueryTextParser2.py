"""
mongoQueryTextParser.py

A textual MongoDB query parser using pyparsing,
converting expressions like:
    role IN ["admin", "superuser"] AND (age >= 18 OR age TYPE int)
to MongoDB query dicts with no ParseResults artifacts.
"""

import pyparsing as pp
import pyparsing.common as ppc


def _make_single_condition(field: str, op: str, value):
    op_map = {
        "==": "$eq",
        "!=": "$ne",
        ">": "$gt",
        ">=": "$gte",
        "<": "$lt",
        "<=": "$lte",
        "IN": "$in",
        "NOT IN": "$nin",
        "ALL": "$all",
        "SIZE": "$size",
        "REGEX": "$regex",
        "TYPE": "$type",
        "EXISTS": "$exists",
        "MOD": "$mod",
        "TEXT": "$text"
    }
    op_upper = op.upper()

    if op_upper == "TEXT":
        return {"$text": {"$search": value}}

    if op_upper == "EXISTS":
        return {field: {"$exists": bool(value)}}
    if op_upper in op_map:
        return {field: {op_map[op_upper]: value}}
    raise ValueError(f"Unknown operator: {op}")


def _merge_conditions(logical_op: str, condition_list: list) -> dict:
    logical_op = logical_op.upper()
    if len(condition_list) == 1:
        return condition_list[0]
    if logical_op == "AND":
        return {"$and": condition_list}
    elif logical_op == "OR":
        return {"$or": condition_list}
    raise ValueError(f"Unsupported logical operator: {logical_op}")


def _apply_not(condition: dict) -> dict:
    if len(condition) == 1:
        (field_key, field_val), = condition.items()
        if isinstance(field_val, dict):
            return {field_key: {"$not": field_val}}
    return {"$not": condition}


def parse_mongo_query(query_str: str) -> dict:
    AND_ = pp.Keyword("AND", caseless=True)
    OR_ = pp.Keyword("OR", caseless=True)
    NOT_ = pp.Keyword("NOT", caseless=True)

    comparison_ops = pp.MatchFirst([
        pp.Keyword("NOT IN", caseless=True),
        pp.Keyword("IN", caseless=True),
        pp.Keyword("==", caseless=True),
        pp.Keyword("!=", caseless=True),
        pp.Keyword(">=", caseless=True),
        pp.Keyword("<=", caseless=True),
        pp.Keyword(">", caseless=True),
        pp.Keyword("<", caseless=True),
        pp.Keyword("ALL", caseless=True),
        pp.Keyword("SIZE", caseless=True),
        pp.Keyword("REGEX", caseless=True),
        pp.Keyword("TYPE", caseless=True),
        pp.Keyword("EXISTS", caseless=True),
        pp.Keyword("MOD", caseless=True),
    ])("op")

    field_name = pp.Word(pp.alphas + "_", pp.alphanums + "._")("field")

    string_value = (
            pp.QuotedString('"', escChar='\\', unquoteResults=True)
            ^ pp.QuotedString("'", escChar='\\', unquoteResults=True)
    ).setParseAction(lambda t: t[0])

    number_value = ppc.number().setParseAction(lambda t: t[0])

    bool_value = (
            pp.Keyword("true", caseless=True).setParseAction(lambda: True)
            | pp.Keyword("false", caseless=True).setParseAction(lambda: False)
    )

    lbrack = pp.Literal("[").suppress()
    rbrack = pp.Literal("]").suppress()
    array_content = pp.Optional(pp.delimitedList(number_value | string_value | bool_value))
    array_value = (lbrack + array_content + rbrack).setParseAction(lambda t: list(t))

    value_expr = array_value | number_value | string_value | bool_value

    condition = pp.Group(
        field_name +
        comparison_ops +
        pp.Optional(value_expr("val"), default=True)
    )

    def condition_parse_action(toks):
        data = toks[0]
        field = data["field"]
        op = data["op"]
        val = data["val"] if "val" in data else True
        return _make_single_condition(field, op, val)

    condition.setParseAction(condition_parse_action)

    expr = pp.Forward()
    factor = condition | (pp.Suppress("(") + expr + pp.Suppress(")"))

    expr <<= pp.infixNotation(
        factor,
        [
            (NOT_, 1, pp.opAssoc.RIGHT, lambda t: [_apply_not(t[0][1])]),
            (AND_, 2, pp.opAssoc.LEFT, lambda t: [_merge_conditions("AND", t[0][0::2])]),
            (OR_, 2, pp.opAssoc.LEFT, lambda t: [_merge_conditions("OR", t[0][0::2])]),
        ]
    )

    parsed = expr.parseString(query_str, parseAll=True)
    return parsed[0]


def _normalize(value):
    import pyparsing as pp
    if isinstance(value, pp.ParseResults):
        # convert to a normal list
        value_list = value.asList()
        # if it's a single-element list, just return that one element
        if len(value_list) == 1:
            return _normalize(value_list[0])
        else:
            return [_normalize(v) for v in value_list]
    elif isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_normalize(x) for x in value]
    else:
        return value


def parse_mongo_query_with_normalization(query_str):
    # ... do your parse as before ...
    AND_ = pp.Keyword("AND", caseless=True)
    OR_ = pp.Keyword("OR", caseless=True)
    NOT_ = pp.Keyword("NOT", caseless=True)

    comparison_ops = pp.MatchFirst([
        pp.Keyword("TEXT", caseless=True),
        pp.Keyword("NOT IN", caseless=True),
        pp.Keyword("IN", caseless=True),
        pp.Keyword("==", caseless=True),
        pp.Keyword("!=", caseless=True),
        pp.Keyword(">=", caseless=True),
        pp.Keyword("<=", caseless=True),
        pp.Keyword(">", caseless=True),
        pp.Keyword("<", caseless=True),
        pp.Keyword("ALL", caseless=True),
        pp.Keyword("SIZE", caseless=True),
        pp.Keyword("REGEX", caseless=True),
        pp.Keyword("TYPE", caseless=True),
        pp.Keyword("EXISTS", caseless=True),
        pp.Keyword("MOD", caseless=True),
    ])("op")

    field_name = pp.Word(pp.alphas + "_", pp.alphanums + "._")("field")

    string_value = (
            pp.QuotedString('"', escChar='\\', unquoteResults=True)
            ^ pp.QuotedString("'", escChar='\\', unquoteResults=True)
    ).setParseAction(lambda t: t[0])

    number_value = ppc.number().setParseAction(lambda t: t[0])

    bool_value = (
            pp.Keyword("true", caseless=True).setParseAction(lambda: True)
            | pp.Keyword("false", caseless=True).setParseAction(lambda: False)
    )

    lbrack = pp.Literal("[").suppress()
    rbrack = pp.Literal("]").suppress()
    array_content = pp.Optional(pp.delimitedList(number_value | string_value | bool_value))
    array_value = (lbrack + array_content + rbrack).setParseAction(lambda t: list(t))

    value_expr = array_value | number_value | string_value | bool_value

    condition = pp.Group(
        field_name +
        comparison_ops +
        pp.Optional(value_expr("val"), default=True)
    )

    def condition_parse_action(toks):
        data = toks[0]
        field = data["field"]
        op = data["op"]
        val = data["val"] if "val" in data else True
        return _make_single_condition(field, op, val)

    condition.setParseAction(condition_parse_action)

    expr = pp.Forward()
    factor = condition | (pp.Suppress("(") + expr + pp.Suppress(")"))

    expr <<= pp.infixNotation(
        factor,
        [
            (NOT_, 1, pp.opAssoc.RIGHT, lambda t: [_apply_not(t[0][1])]),
            (AND_, 2, pp.opAssoc.LEFT, lambda t: [_merge_conditions("AND", t[0][0::2])]),
            (OR_, 2, pp.opAssoc.LEFT, lambda t: [_merge_conditions("OR", t[0][0::2])]),
        ]
    )

    parsed = expr.parseString(query_str, parseAll=True)
    raw_dict = parsed[0]
    return _normalize(raw_dict)


if __name__ == "__main__":
    test_queries = [
        'role IN ["admin", "superuser"] AND (age >= 18 OR age TYPE int)',
        'category NOT IN ["food", "drink"]',
        '(age == 21 AND name == "test") OR age > 30',
        'NOT city EXISTS AND tags SIZE 3',
        'price MOD [4, 0]',
        'title REGEX "^MongoDB"',
        'status EXISTS false'
    ]
    for q in test_queries:
        print("Query:", q)
        try:
            mongo_filter = parse_mongo_query_with_normalization(q)
            print("MongoDB Filter:", mongo_filter)
        except pp.ParseException as pe:
            print("Parse Error:", pe)
        except Exception as e:
            print("Error:", e)
        print("-" * 60)
