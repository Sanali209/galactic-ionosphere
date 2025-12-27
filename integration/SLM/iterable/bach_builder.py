from tqdm import tqdm


class BatchBuilder:
    def __init__(self, list, bach_size=1000):
        self.bach_size = bach_size
        self.list = list
        self.bach: dict = {}
        self.on_bach_end_callback = None
        self.build()

    def build(self):
        self.bach = {}
        bach_counter = 0
        for item in self.list:
            bach_list = self.bach.get(bach_counter, [])
            bach_list.append(item)
            self.bach[bach_counter] = bach_list
            if len(bach_list) >= self.bach_size:
                bach_counter += 1

    def for_each(self, func):
        tqdm_total = len(self.list)
        progress = tqdm(total=tqdm_total)
        for val in self.bach.values():
            for item in val:
                progress.update(1)
                func(item)
            if self.on_bach_end_callback is not None:
                self.on_bach_end_callback()
