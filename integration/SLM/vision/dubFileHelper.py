


class DuplicateFindHelper:

    def sort_images_by_futures(self, images: list[str]) -> list[str]:
        images_futures = {}
        sorted_list = []
        for image in tqdm(images, desc='encoding images'):
            images_futures[image] = self.GetCNNEncoding(image)

        future_lenth = len(images_futures[images[0]])

        annoi_index = AnnoyIndex(future_lenth, 'euclidean')
        for i, v in tqdm(zip(range(len(images_futures)), images_futures.values())):
            annoi_index.add_item(i, v)

        annoi_index.build(1000)

        distances = np.zeros((len(images), len(images)))
        for a in tqdm(range(len(images)), desc='computing distances'):
            for b in range(a + 1, len(images)):
                distance = annoi_index.get_distance(a, b)
                distances[a][b] = distance
                distances[b][a] = distance

        minets = 10
        logger.info(f'start solving tsp with simulated annealing for {minets} minets')
        start_time = time.time()
        ordering, total_distance = solve_tsp_simulated_annealing(distances, max_processing_time=minets * 60)
        logger.info(f'solved tsp with simulated annealing in {time.time() - start_time} seconds')

        for i in tqdm(ordering):
            sorted_list.append(images[i])

        return sorted_list

    def sort_images_by_futures_base(self, images: list[str]) -> list[str]:
        from SLM.vision.imagetotensor.CNN_Encoding import ImageToCNNTensor
        im_data_tensors = ImageToCNNTensor()
        images_futures = {}
        sorted_list = []
        unsorted_list = images.copy()
        for image in tqdm(images, desc='encoding images'):
            images_futures[image] = im_data_tensors.get_tensor_from_path(image)

        future_lenth = len(images_futures[images[0]])

        annoi_index = AnnoyIndex(future_lenth, 'euclidean')
        for i, v in tqdm(zip(range(len(images_futures)), images_futures.values())):
            annoi_index.add_item(i, v)

        annoi_index.build(1000)
        image = unsorted_list.pop(0)
        sorted_list.append(image)
        progress = tqdm(total=len(images))
        while len(unsorted_list) > 0:
            progress.update(1)
            shortest_distance = 0
            shortest_image = None
            first = True
            # distance
            for image2 in unsorted_list:
                distance = annoi_index.get_distance(images.index(image), images.index(image2))
                if first:
                    shortest_distance = distance
                    shortest_image = image2
                    first = False
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_image = image2
            if shortest_image is None:
                break
            image = shortest_image
            unsorted_list.remove(shortest_image)
            sorted_list.append(shortest_image)

        return sorted_list

        return len(sortet1.symmetric_difference(sortet2)) == 0

