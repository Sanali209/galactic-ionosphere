class ItemIndexer:
    def __init__(self):
        self.field_indexers = {}
        self.field_indexers_done = {}
        self.shared_data = {}

    def add_field_indexer(self, field_indexer: 'ItemFieldIndexer'):
        self.field_indexers[field_indexer.fieldName] = field_indexer
        return self

    # overide | symbol
    def __or__(self, other):
        self.add_field_indexer(other)
        return self

    def index(self, item, reindex=False):
        while len(self.field_indexers) > 0:
            field_name, field_indexer = self.field_indexers.popitem()
            if not field_indexer.enabled or field_indexer.run_on_dependent:
                continue
            # invoce dependent fields

            need_index = field_indexer.isNeedIndex(self, item)
            need_index = need_index or reindex
            if need_index:
                res = field_indexer.invoce_dependent_fields(self, item, reindex)
                field_indexer.index(self, item, need_index)
            self.field_indexers_done[field_name] = field_indexer

        # copy done to indexers and clear done
        self.field_indexers = self.field_indexers_done
        self.field_indexers_done = {}

    def invoce_dependent_fields(self, item, field_name, reindex):
        if field_name in self.field_indexers:
            field = self.field_indexers.pop(field_name)
            need_index = field.isNeedIndex(self, item)
            need_index = need_index or reindex
            res = None
            if need_index:
                field.invoce_dependent_fields(self, item, reindex)
                res = field.index(self, item, need_index)
            self.field_indexers_done[field_name] = field
            return res

    def reset_shared_data(self):
        self.shared_data = {}


class ItemFieldIndexer:
    def __init__(self, enabled=True):
        self.fieldName = "default"
        self.enabled = enabled
        self.dependent_fields = []
        self.run_on_dependent = False

    def mark_as_indexed(self, item, parent_indexer):
        parent_indexer.shared_data["item_indexed"] = True

    def isNeedIndex(self, parent_indexer: ItemIndexer, item):
        return True

    def invoce_dependent_fields(self, parent_indexer: ItemIndexer, item, need_index: bool):
        for field_name in self.dependent_fields:
            parent_indexer.invoce_dependent_fields(item, field_name, need_index)

    def index(self, parent_indexer: ItemIndexer, item, need_index: bool):
        pass
