from abc import ABC

import loguru
from annoy import AnnoyIndex
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.iterable.bach_builder import BatchBuilder


class ResultItem:
    def __init__(self):
        self.data_item = None
        self.distance = 0


class ResultGroup:
    def __init__(self):
        self.data_item = None
        self.results: list[ResultItem] = []


class VectorDB:
    table_initializers = {}

    tables = {}

    @staticmethod
    def get_path():
        config = Allocator.config.mongoConfig
        return config.path + "/vector_db"

    @staticmethod
    def register_pref(name, vector_size, vector_function, metric_name):
        """
        :param name:
        :param vector_size:
        :param vector_function:
        :param metric_name:  "angular", "euclidean", "manhattan", "hamming", or "dot"
        :return:
        """
        VectorDB.table_initializers[name] = (vector_function, metric_name, vector_size)

    @staticmethod
    def get_pref(name):
        if name not in VectorDB.tables:
            vc_function, metric_name, vector_size = VectorDB.table_initializers[name]
            VectorDB.tables[name] = VectorDBPreferences(name, vc_function, metric_name, vector_size)
        return VectorDB.tables[name]


class VectorDBPreferences:
    """
    :type name: str
    :type vector_function: function
    :type metric_name: str # "angular", "euclidean", "manhattan", "hamming", or "dot"
    :type vector_size: int
    """

    def __init__(self, name, vector_function, metric_name, vector_size):
        """

        :param name:
        :param vector_function:
        :param metric_name:  "angular", "euclidean", "manhattan", "hamming", or "dot"
        :param vector_size:
        """
        self.name = name
        self.vector_function = vector_function
        self.metric_name = metric_name  # "angular", "euclidean", "manhattan", "hamming", or "dot"
        self.vector_size = vector_size


# todo: add swich to multitread
class SearchScope(ABC):
    def __init__(self, db_table):
        self.index = AnnoyIndex(db_table.vector_size, db_table.metric_name)
        self.vector_function = db_table.vector_function
        self.items_set = []
        self.vector_store = []
        self.index_dict = {}
        vect_items = self.get_items_to_vectorization()
        bach_b = BatchBuilder(vect_items, 16)
        self.vector_size = db_table.vector_size

        def vectorize_item(item):
            try:
                vector = self.vector_function(item)
                if vector is None:
                    return None
                if len(vector) != self.vector_size:
                    return None
                self.index_dict.setdefault(item, {}).setdefault("vector", vector)
                self.items_set.append(item)
            except Exception as e:
                print("Error in vectorization")
                loguru.logger.exception(e)
                return None

        for batch in tqdm(bach_b.bach.values(), desc="Indexing"):
            #with ThreadPoolExecutor(6) as executor:
                #list(executor.map(vectorize_item, batch))
            for item in batch:
                vectorize_item(item)
        # map items
        for index, item in tqdm(enumerate(self.items_set), desc="Building index"):
            self.index_dict[item]["index"] = index
            try:

                self.index.add_item(index, self.index_dict[item]["vector"])
            except Exception as e:
                loguru.logger.exception(e)
        self.index.build(20)

    def get_items_to_vectorization(self):
        return []

    def search(self, search_term, limit=10, distance_threshold=0) -> ResultGroup:
        search_vector = self.vector_function(search_term)
        try:
            search_term_index = self.index_dict[search_term]["index"]
        except:
            search_term_index = None
        return self.search_by_vector(search_vector, limit, distance_threshold, search_term_index)

    def search_by_vector(self, search_vector, limit=10,
                         distance_threshold=0, search_term_index=None) -> ResultGroup:
        if search_term_index is not None:
            limit += 1
        result = self.index.get_nns_by_vector(search_vector, limit, include_distances=True)
        ret_result = ResultGroup()
        if search_term_index is not None:
            ret_result.data_item = self.items_set[search_term_index]
            try:
                index = result[0].index(search_term_index)
                value = result[0][index]
                result[0].remove(value)
                value = result[1][index]
                result[1].remove(value)
            except:
                pass

        for index, distance in zip(result[0], result[1]):
            if distance > distance_threshold != 0:
                continue
            sim_res = ResultItem()
            sim_res.data_item = self.items_set[index]
            sim_res.distance = distance
            ret_result.results.append(sim_res)
        return ret_result

    def find_dubs(self, limit: int = 10, distance_threshold=0) -> list[ResultGroup]:
        items_list: list = self.items_set
        list_of_dubs = []
        progress_bar = tqdm(total=len(items_list))

        def find_top_and_append(item):
            progress_bar.update(1)
            try:
                find_result_group = self.search(item, limit, distance_threshold)
                if len(find_result_group.results) > 0:
                    return find_result_group
                    #list_of_dubs.append(find_result_group)
            except Exception as e:
                print("Error in find_dubs")
                print(e)

        for item in items_list:
            res = find_top_and_append(item)
            if res is not None:
                list_of_dubs.append(res)

        #with ThreadPoolExecutor() as executor:
        #list(executor.map(find_top_and_append, items_list))

        return list_of_dubs


class SearchScopeList(SearchScope):
    def __init__(self, db_table, items_list):
        self.items_list = items_list
        super().__init__(db_table)

    def get_items_to_vectorization(self):
        return self.items_list
