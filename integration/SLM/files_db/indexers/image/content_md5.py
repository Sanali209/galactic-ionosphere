

from SLM.indexerpyiplain.idexpyiplain import ItemFieldIndexer, ItemIndexer


def mark_as_indexed_by_indexer(data, field_indexer_name, parent_indexer):
    if 'indexed_by' not in data:
        data['indexed_by'] = []
    data['indexed_by'].append(field_indexer_name)
    parent_indexer.shared_data["item_indexed"] = True


class files_db_indexer(ItemFieldIndexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fieldName = "files_db_indexer"

    def isNeedIndex(self, parent_indexer, item):
        if self.fieldName in item.get('indexed_by', []):
            return False
        return True

    def mark_as_indexed(self,item, parent_indexer):
        mark_as_indexed_by_indexer(item, self.fieldName, parent_indexer)


