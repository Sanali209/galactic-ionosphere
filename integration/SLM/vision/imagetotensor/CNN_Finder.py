"""
.. todo::
    1. remove dependency from dask
    2. remove dependency from `imagededup <https://github.com/idealo/imagededup>`_
"""

# todo change inputh formath end exporth format
# todo implement interfaces for unification

from annoy import AnnoyIndex
from tqdm import tqdm
from dask.distributed import Client
from SLM.files_data_cache.tensor import Embeddings_cache
from SLM.iterable.bach_builder import BatchBuilder
from concurrent.futures import ThreadPoolExecutor

from SLM.vision.imagetotensor.backends.mobile_net_v3 import CNN_encoder_ModileNetv3_Small


class FindResultItem:
    def __init__(self):
        self.path = ""
        self.distance = 0


class FindResultGroup:
    def __init__(self):
        self.path = ""
        self.results = []
        self.none_tensor = False


class CNN_Dub_Finder:
    """
    Class for find dubs in image dataset
    todo implement interface for unification of search classes
    """
    def __init__(self):
        self.path_to_index = {}
        self.name = "CNN_Encoded_Top_Finder"
        self.encodingIndex = {}
        self.annoy_index:AnnoyIndex = None
        self.tensor_format = CNN_encoder_ModileNetv3_Small.format
        self.metric: str = "angular" #"angular", "euclidean", "manhattan", "hamming", or "dot"
        self.DbImageObjectItems = []
        """list of search objects. Need for convert index number to ref_object"""
        self.dub_find_neighbors = 10
        self.threshold_map = {
            'modilenetv3': {'angular': 0.6, 'euclidean': 0.5},
        }

    def _BuildIndex_dusk(self, image_paths: list[str]):
        # dushbord on http://localhost:8787/
        # not have sense to use it in this case computation time less transfer time
        self.DbImageObjectItems = image_paths
        first = image_paths[0]
        tensor_cache = Embeddings_cache([self.tensor_format])
        first_np_tensor = tensor_cache.get_by_path(first, self.tensor_format)
        tensor_length = len(first_np_tensor)
        self.annoy_index = AnnoyIndex(tensor_length, self.metric)
        # dushbord on http://localhost:8787/
        dusk_client = Client(processes=False)
        bach_b = BatchBuilder(image_paths, 64)

        def get_embedding(path, cache, format):
            return cache.get_by_path(path, format)

        embeddings = []
        for buch in tqdm(bach_b.bach.values()):
            futures = dusk_client.map(
                get_embedding, buch,
                [tensor_cache] * len(buch),
                [self.tensor_format] * len(buch))
            dusk_client.gather(futures)
            for future in futures:
                res = future.result()
                embeddings.append(res)

        for index, dbitem, embeding in tqdm(zip(range(len(image_paths)), image_paths, embeddings),
                                            total=len(image_paths)):
            self.annoy_index.add_item(index, embeding)

        self.annoy_index.build(100)

    def BuildIndex(self, image_paths: list[str]):
        self.DbImageObjectItems = image_paths
        self.path_to_index = {}
        first = image_paths[0]
        tensor_cache = Embeddings_cache([self.tensor_format])
        first_np_tensor = tensor_cache.get_by_path(first, self.tensor_format)
        tensor_length = len(first_np_tensor)
        self.tensor_length = tensor_length
        self.annoy_index = AnnoyIndex(tensor_length, self.metric)
        for index, dbitem in tqdm(zip(range(len(image_paths)), image_paths),
                                  total=len(image_paths)):
            tensor = tensor_cache.get_by_path(dbitem, self.tensor_format)
            if tensor is None:
                print("Error: tensor is None")
                continue
            if len(tensor) != tensor_length:
                print("Error: tensor length is not equal")
                continue
            self.path_to_index[dbitem] = index


            self.annoy_index.add_item(index, tensor)
        self.annoy_index.build(100)

    def FindTop(self, find_item_path: str, top_count=10, distance_threshold=0) -> FindResultGroup:
        tensor_cache = Embeddings_cache([self.tensor_format])
        find_result_group = FindResultGroup()
        find_result_group.path = find_item_path
        template_tensor_np = tensor_cache.get_by_path(find_item_path, self.tensor_format)
        if template_tensor_np is None:
            find_result_group.none_tensor = True
            return find_result_group
        path_index = self.path_to_index.get(find_item_path, None)
        if path_index is not None:
            top_count += 1
        if len(template_tensor_np) != self.tensor_length:
            print("Error: tensor length is not equal")
            return find_result_group
        result_indexes = self.annoy_index.get_nns_by_vector(template_tensor_np, top_count, include_distances=True)
        for index, distance in zip(result_indexes[0], result_indexes[1]):
            if path_index is not None and path_index == index:
                continue
            if distance <= distance_threshold or distance_threshold == 0:
                find_result_item = FindResultItem()
                image_path = self.DbImageObjectItems[index]
                find_result_item.path = image_path
                find_result_item.distance = distance
                find_result_group.results.append(find_result_item)
        return find_result_group

    def FindDubs(self, image_paths: list[str], distance_threshold=0) -> list:
        if distance_threshold == -1:
            try:
                distance_threshold = self.threshold_map[self.tensor_format][self.metric]
            except KeyError:
                distance_threshold = 0
        self.BuildIndex(image_paths)
        list_of_dubs = []

        progress_bar = tqdm(total=len(image_paths))

        def find_top_and_append(dbitem):
            find_result_group = self.FindTop(dbitem, self.dub_find_neighbors, distance_threshold)
            if len(find_result_group.results) > 0:
                list_of_dubs.append(find_result_group)
            progress_bar.update(1)

        with ThreadPoolExecutor() as executor:
            list(executor.map(find_top_and_append, image_paths))
        return list_of_dubs
