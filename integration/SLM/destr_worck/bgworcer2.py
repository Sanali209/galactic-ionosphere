import json
import os
import time
from threading import Thread
from abc import ABC, abstractmethod
from loguru import logger


class BGTaskState:
    def __init__(self, status="in_progress", progress=0, progress_message="", result=None):
        self.status = status
        self.progress = progress
        self.progress_message = progress_message
        self.result = result

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data):
        return BGTaskState(**data)


class TaskRepository:
    """
    Репозиторий для регистрации и восстановления классов задач.
    Позволяет добавлять пользовательские задачи и восстанавливать их из сохраненного состояния.
    """
    _registry = {}

    @classmethod
    def register_task(cls, task_cls):
        cls._registry[task_cls.__name__] = task_cls

    @classmethod
    def get_task_class(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Task class '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def create_task(cls, task_data):
        task_cls = cls.get_task_class(task_data["class_name"])
        return task_cls.from_dict(task_data)


class BGTask:
    def __init__(self, name, *args, **kwargs):
        self.manager = None
        self.name = name
        self.cancel_names = []
        self.exclude_names = []
        self.args = args
        self.kwargs = kwargs
        self.done_callback = None
        self.state = BGTaskState()
        self.queue_name = "default"
        self.generator = self.task_function()

    def to_dict(self):
        return {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "state": self.state.to_dict(),
            "args": self.args,
            "kwargs": self.kwargs,
            "queue_name": self.queue_name,
        }

    @staticmethod
    def from_dict(data):
        task_cls = TaskRepository.get_task_class(data["class_name"])
        task = task_cls(data["name"], *data["args"], **data["kwargs"])
        task.state = BGTaskState.from_dict(data["state"])
        task.queue_name = data["queue_name"]
        return task

    def task_function(self):
        yield

    def run(self):
        if self.state.status in ["done", "cancel", "error"]:
            return
        try:
            next(self.generator)
            if self.state.progress >= 100:
                self.set_done()
        except StopIteration:
            self.set_done()
        except Exception as e:
            self.set_error(e)

    def set_done(self, result=None):
        self.state.status = "done"
        self.state.result = result
        self.fire_done()

    def set_error(self, error):
        self.state.status = "error"
        self.state.result = error
        self.fire_done()

    def fire_done(self):
        if self.done_callback:
            self.done_callback(self)


class CustomTask(BGTask):
    def task_function(self):
        for i in range(5):
            time.sleep(1)
            self.state.progress = (i + 1) * 20
            yield


TaskRepository.register_task(BGTask)
TaskRepository.register_task(CustomTask)


class TaskStoreInterface(ABC):
    @abstractmethod
    def save_tasks(self, tasks, queue_name):
        pass

    @abstractmethod
    def load_tasks(self, queue_name):
        pass

    @abstractmethod
    def delete_tasks(self, queue_name):
        pass


class JSONTaskStore(TaskStoreInterface):
    def __init__(self, storage_dir="task_store"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_file_path(self, queue_name):
        return os.path.join(self.storage_dir, f"{queue_name}.json")

    def save_tasks(self, tasks, queue_name):
        file_path = self._get_file_path(queue_name)
        with open(file_path, "w") as f:
            json.dump([task.to_dict() for task in tasks], f)

    def load_tasks(self, queue_name):
        file_path = self._get_file_path(queue_name)
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r") as f:
            tasks_data = json.load(f)
            return [TaskRepository.create_task(task_data) for task_data in tasks_data]

    def delete_tasks(self, queue_name):
        file_path = self._get_file_path(queue_name)
        if os.path.exists(file_path):
            os.remove(file_path)


class TaskQueueSeq:
    def __init__(self, name="default", task_store=None):
        self.name = name
        self.tasks = []
        self.cur_task = None
        self.task_store = task_store
        if self.task_store:
            self.tasks = self.task_store.load_tasks(self.name)

    def is_empty(self):
        return len(self.tasks) == 0

    def add_task(self, task):
        self.tasks.append(task)
        self._save_tasks()

    def do_tasks(self):
        if self.is_empty():
            return
        self.cur_task = self.tasks.pop(0)
        self.cur_task.run()
        if self.cur_task.state.status != "done":
            self.tasks.append(self.cur_task)
        self.cur_task = None
        self._save_tasks()

    def _save_tasks(self):
        if self.task_store:
            self.task_store.save_tasks(self.tasks, self.name)


class BGWorker:
    def __init__(self, task_store=None):
        self.tasks_queues = {}
        self.run_t = True
        self.task_store = task_store or JSONTaskStore()
        self.add_queue("default")
        self.work_thread = Thread(target=self.tusks_execute)
        self.work_thread.start()

    def add_queue(self, name):
        self.tasks_queues[name] = TaskQueueSeq(name, self.task_store)

    def add_task(self, task):
        task.manager = self
        if task.queue_name not in self.tasks_queues:
            self.add_queue(task.queue_name)
        self.tasks_queues[task.queue_name].add_task(task)

    def tusks_execute(self):
        while self.run_t:
            if all(queue.is_empty() for queue in self.tasks_queues.values()):
                time.sleep(1)
            for queue in self.tasks_queues.values():
                queue.do_tasks()

    def stop(self):
        self.run_t = False


if __name__ == "__main__":
    task_store = JSONTaskStore()
    worker = BGWorker(task_store)

    # Добавление задач
    task1 = BGTask("Task 1")
    task2 = CustomTask("Custom Task")

    worker.add_task(task1)
    worker.add_task(task2)

    time.sleep(5)  # Позволяем задачам выполняться

    worker.stop()

    # Проверяем восстановление
    worker = BGWorker(task_store)
    worker.stop()
