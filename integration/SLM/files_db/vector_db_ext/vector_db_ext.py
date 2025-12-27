

from SLM.appGlue.DAL.datalist2 import MongoDataModel, DataViewCursor
from SLM.appGlue.core import Allocator

from SLM.vector_db.vector_db import SearchScope


class SearchScopeMongoDb(SearchScope):
    def __init__(self, query, db_table):
        config = Allocator.config.mongoConfig
        db_name = config.database_name
        self.data_model = MongoDataModel("mongodb://localhost:27017",
                                         db_name,
                                         "collection_records")
        self.data_list_cursor = DataViewCursor(self.data_model)
        self.scope_query = query
        super().__init__(db_table)

    def get_items_to_vectorization(self):
        self.data_list_cursor.set_specification(self.scope_query)
        return self.data_list_cursor.get_filtered_data(all_pages=True)
