from loguru import logger
from tqdm import tqdm

from SLM.appGlue.DAL.DAL import GlueDataConverter, ForwardConverter
from SLM.appGlue.DesignPaterns.specification import Specification, AllSpecification
from pymongo import MongoClient

from abc import ABC, abstractmethod


# todo nintegrate to annotation model
# todo replase previos version of datalist

class AbstractDataModel(ABC):
    @abstractmethod
    def attach(self, observer):
        pass

    @abstractmethod
    def detach(self, observer):
        pass

    @abstractmethod
    def _notify(self, change_type, item=None):
        pass

    @abstractmethod
    def append(self, item):
        pass

    @abstractmethod
    def remove(self, item):
        pass

    @abstractmethod
    def clear(self):
        pass


class AbstractDataQuery(ABC):
    @abstractmethod
    def get_by_query(self, skip=0, limit=0, sort="default"):
        pass

    @abstractmethod
    def obj_to_query(self, obj):
        pass

    @abstractmethod
    def count_all(self):
        pass


class DataObserver:

    def update(self, data_model, change_type, item=None):
        pass


class DataQuery(AbstractDataQuery):
    def __init__(self, data_model: 'DataListModel'):
        self._data_model = data_model
        self._specification = AllSpecification()
        self._filtered_data = None
        self._last_version = -1

    def _chech_and_refresh(self):
        current_version = self._data_model._version
        if current_version != self._last_version:
            self._last_version = current_version
            self._filtered_data = None

    def get_by_query(self, skip=0, limit=0, sort=None,sort_algs=None):
        self._chech_and_refresh()

        if self._filtered_data is None:
            self._filtered_data = self._data_model._data
            if self._specification is not None:
                self._filtered_data = []
                for item in tqdm(self._data_model._data, desc="Filtering data"):
                    if self._specification.is_satisfied_by(item):
                        self._filtered_data.append(item)
            if sort is not None:
                self._filtered_data = sort_algs[sort](self._filtered_data)

        ret_list = self._filtered_data[skip:]
        if limit > 0:
            ret_list = ret_list[:limit]
        return ret_list

    def obj_to_query(self, obj):
        self._specification = obj
        self._filtered_data = None

    def count_all(self):
        return len(self.get_by_query())


class DataListModelBase(AbstractDataModel):
    def __init__(self):
        self._observers = []
        self.dataQuery = DataQuery(self)

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def _notify(self, change_type, item=None):
        for observer in self._observers:
            observer.list_update(self, change_type, item)

    def append(self, item):
        self._notify("add", item)

    def extend(self, items):
        self._notify("refresh", None)

    def refresh_view(self):
        self._notify("refresh", None)

    def remove(self, item):
        self._notify("remove", item)

    def clear(self):
        self._notify("clear")


class DataListModel(DataListModelBase):
    def __init__(self):
        super().__init__()
        self._data = []
        self._version = 0

    def append(self, item):
        if item in self._data:
            return
        self._version += 1
        self._data.append(item)
        super().append(item)

    def extend(self, items):
        self._version += 1
        items = list(set(items))
        add_items = [item for item in items if item not in self._data]
        self._data.extend(add_items)
        super().extend(items)

    def remove(self, item):
        self._version += 1
        if item in self._data:
            self._data.remove(item)
        super().remove(item)

    def clear(self):
        self._version += 1
        self._data.clear()
        super().clear()

    def exist(self, file):
        seti = set(self._data)
        if file in seti:
            return True
        return False

    def __contains__(self, item):
        return self.exist(item)


class DataViewCursor(DataObserver):
    def __init__(self, data_model: DataListModel, data_converter: GlueDataConverter = None):
        super().__init__()
        self.max_items = None
        self._data_model = data_model
        self._current_index = 0
        data_model.attach(self)
        self.child_observers = []
        self.data_converter = data_converter
        if self.data_converter is None:
            self.data_converter = ForwardConverter()
        self.items_per_page = 100
        self.current_page = 0
        self.max_page = 0
        self.sort = None
        self._specification = AllSpecification()
        self.all_items_count()
        self._view_model = []
        self.sort_alg = {"default": self.sort_def}
        self.on_pre_refresh = None

    def sort_def(self,  list):

        return list.sort(key=lambda x: x)

    def __iter__(self):
        for item in self.get_filtered_data():
            yield item

    def __str__(self):
        return str(self.get_filtered_data())

    def __repr__(self):
        return str(self.get_filtered_data())

    def get_item_on_index(self, index):
        return self.get_filtered_data()[index]

    def get_index_of_item(self, item):
        if item in self.get_filtered_data():
            return self.get_filtered_data().index(item)
        else:
            return -1

    def attach(self, observer):
        self.child_observers.append(observer)

    def set_specification(self, spec: Specification):
        self._specification = spec
        self._data_model.dataQuery.obj_to_query(self._specification)
        self.current_page = 0
        self._current_index = 0
        self.refresh()

    def get_filtered_data(self, skip=0, limit=0, all_pages=False):

        if skip == 0:
            skip = self.current_page * self.items_per_page
        if limit == 0:
            limit = self.items_per_page
        if all_pages:
            skip = 0
            limit = 0
        list_ = self._data_model.dataQuery.get_by_query(skip, limit, self.sort,self.sort_alg)
        if not isinstance(self.data_converter, ForwardConverter):
            return [self.data_converter.Convert(item) for item in list_]
        return list_

    def get_filtered_item(self, index):
        return self.data_converter.Convert(self.get_filtered_data()[index])

    def all_items_count(self):
        try:
            self.max_items = self._data_model.dataQuery.count_all()
            if self.items_per_page > 0:
                self.max_page = self.max_items // self.items_per_page
            return self.max_items
        except Exception as e:
            logger.error(f"Error in DataViewCursor.all_items_count: {e}")
            return 0

    def get_current_item(self):
        filtered_data_list = self.get_filtered_data(self._current_index, 1)
        if len(filtered_data_list) > 0:
            return filtered_data_list[0]
        else:
            return None

    def move_next(self):
        if self._current_index < self.max_items - 1:
            self._current_index += 1
        else:
            self._current_index = 0

    def move_previous(self):
        if self._current_index > 0:
            self._current_index -= 1
        else:
            self._current_index = self.max_items - 1

    def move_to_start(self):
        self._current_index = 0

    def move_to_end(self):
        self._current_index = self.max_items - 1

    def move_to_index(self, index):
        if index < len(self.max_items):
            self._current_index = index
        elif len(self.max_items) > 0:
            self._current_index = self.max_items - 1
        else:
            self._current_index = 0

    def page_next(self):
        if self.current_page < self.max_page:
            self.current_page += 1
            self.refresh()

    def page_previous(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh()

    def refresh(self):
        if self.on_pre_refresh is not None:
            self.on_pre_refresh(self)
        for observer in self.child_observers:
            observer.list_update(self, "refresh")

    def in_view(self, item):
        """
        Check if item in current view
        :param item:
        :return:
        """
        if len(self._view_model) < self.items_per_page:
            self._view_model = self.get_filtered_data(self.current_page * self.items_per_page, self.items_per_page)
        return item in self._view_model

    def list_update(self, data_model, change_type, item=None):
        if change_type == "add":
            if self.in_view(item):
                for observer in self.child_observers:
                    observer.list_update(self, change_type, self.data_converter.Convert(item))
            else:
                for observer in self.child_observers:
                    observer.list_update(self, "add_no_in_view", self.data_converter.Convert(item))
            return None

        if change_type == "remove":
            self._view_model = self.get_filtered_data(self.current_page * self.items_per_page, self.items_per_page)
            #self._data_model.dataQuery.obj_to_query(self._specification)
        elif change_type == "clear":
            #self._data_model.dataQuery.obj_to_query(self._specification)
            self._current_index = 0
            self.current_page = 0
            self._view_model = self.get_filtered_data(self.current_page * self.items_per_page, self.items_per_page)
        elif change_type == "refresh":
            self._current_index = 0
            self._view_model = self.get_filtered_data(self.current_page * self.items_per_page, self.items_per_page)
        #self.all_items_count()
        # Уведомление дочерних наблюдателей
        for observer in self.child_observers:
            observer.list_update(self, change_type, self.data_converter.Convert(item))

    def append(self, item):
        self._data_model.append(self.data_converter.ConvertBack(item))

    def remove(self, item):
        self._data_model.remove(self.data_converter.ConvertBack(item))

    def clear(self):
        self._data_model.clear()


class MongoDataModel(DataListModelBase):
    def __init__(self, uri, database, collection):
        super().__init__()
        self._client = MongoClient(uri)
        self._database = self._client[database]
        self._collection = self._database[collection]
        self.dataQuery = MongoDataQuery(self)

    def append(self, item):
        result = self._collection.insert_one(item)
        super().append(item)
        return result.inserted_id

    def remove(self, item):
        query = {"_id": item["_id"]}
        result = self._collection.delete_one(query)
        if result.deleted_count > 0:
            super().remove(item)

    def clear(self):
        result = self._collection.delete_many({})
        if result.deleted_count > 0:
            super().clear()


#add caching copabilities
class MongoDataQuery(AbstractDataQuery):
    def __init__(self, data_model: MongoDataModel):
        self._data_model = data_model
        self._filter = {}
        self.aggregation_pipline = {}

    def get_by_query(self, skip=0, limit=0, sort=None):
        if sort is None:
            sort = [("_id", 1)]
        query = self._filter
        try:
            cursor = self._data_model._collection.find(query).sort(sort).skip(skip).limit(limit)
        except Exception as e:
            print(e)
            return []

        res = list(cursor)
        return res

    def count_all(self):
        return self._data_model._collection.count_documents(self._filter)

    def obj_to_query(self, obj):
        # if object not dictionary set filter to empty
        if not isinstance(obj, dict):
            self._filter = {}
            return
        if obj is None:
            self._filter = {}
        else:
            self._filter = obj
