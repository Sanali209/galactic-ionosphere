class ChainFunction:
    def __init__(self):
        self.next_f: ChainFunction = None
        self.operation = None
        self.in_dict = {}
        self.pass_truth_in_dict = False

    def set_last(self, last_f: 'ChainFunction'):
        if self.next_f is not None:
            self.next_f.set_last(last_f)
        else:
            self.next_f = last_f

    def __or__(self, other):
        self.set_last(other)
        return self

    def execute_operation(self):
        if self.operation is not None:
            return self.operation(**self.in_dict)
        return self.in_dict

    def run(self, **kwargs):
        self.in_dict = {**kwargs}
        result = self.execute_operation()
        if self.next_f is not None:
            return self.next_f.run(**result)
        return result


class PrintToChainFunction(ChainFunction):
    def __init__(self):
        super().__init__()

        def printTest(message):
            print(message)
            return {'response': 'ok'}

        self.operation = printTest

    def run(self, message="", **kwargs):
        kwargs["message"] = message
        return super().run(**kwargs)


class DictFormatterChainFunction(ChainFunction):
    """
    This class will format dictionary fields

        Sample usage:

        format = DictFormatterChainFunction()
        format.mapping = {"target_field": ["field1", "field2"], "target_field2": ["field3", "field4"]}
        format.validators = {"target_field": lambda val: val}
    """
    def __init__(self):
        super().__init__()
        self.mapping = {}
        self.validators = {}
        self.copy_source = False
        self.verbose = True
        self.operation = self.format

    def format(self, **kwargs):
        ret_dict: dict = {}
        if self.copy_source:
            ret_dict = {**kwargs}
        for dest_key in self.mapping.keys():
            if not ret_dict.__contains__(dest_key):
                source_keys = self.mapping[dest_key]
                for s_key in source_keys:
                    if kwargs.__contains__(s_key):
                        validator = self.validators.get(dest_key, None)
                        if validator is None:
                            validator = lambda val: val
                        val = validator(kwargs[s_key])
                        ret_dict[dest_key] = val
                        break

        return ret_dict


class DictFieldMergeChainFunction(ChainFunction):
    """
    This class will merge fields from input dictionary to one field

    Sample usage:


    merge = DictFieldMergeChainFunction()
    merge.map = {"target_field": ["field1", "field2"], "target_field2": ["field3", "field4"]}
    """

    def __init__(self):

        super().__init__()
        self.map = {}
        self.pass_truth_dict = True
        self.operation = self.collect
        self.verbose = False
        self.splitter:str = "|"
        self.include_source_name = True

    def collect(self, **kwargs):
        ret_dict = {}
        if self.pass_truth_dict:
            ret_dict = {**kwargs}

        for key in self.map:
            summary_str = []
            fields = self.map[key]
            for field in fields:
                if kwargs.__contains__(field):
                    if self.include_source_name:
                        summary_str.append(f"{field}={str(kwargs[field])}")
                    else:
                        summary_str.append(str(kwargs[field]))
                    ret_dict[key] = self.splitter.join(summary_str)

        return ret_dict
