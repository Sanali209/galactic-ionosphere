import os
import uuid
import time
os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
os.environ['APPDATA'] = r"D:\data\ImageDataManager"
os.environ['MONGODB_NAME'] = "files_db"
import grpc
from PIL import Image
from tqdm import tqdm

import task_service_pb2
import task_service_pb2_grpc
import threading

from SLM.files_db.components.File_record_wraper import get_file_record_by_folder, FileRecord


class ATask:
    def __init__(self,  task_name, args,task_id=None):
        self.task_id = task_id
        if task_id is None:
            self.task_id = str(uuid.uuid4())
        self.task_name = task_name
        self.args = args
        self.status = "Task not completed"
        self.result = None
        self.task_complete_callback = None

    def set_task_complete_callback(self, callback):
        self.task_complete_callback = callback

    def setResult(self, result):
        self.result = result
        self.status = "Task completed"
        if self.task_complete_callback is not None:
            self.task_complete_callback(self)

    def __str__(self):
        return f"Task {self.task_id}, name: {self.task_name}, args: {self.args}"


class TaskClient:
    def __init__(self,chanel_address='localhost:50051'):
        self.channel = grpc.insecure_channel(chanel_address)
        self.stub = task_service_pb2_grpc.TaskServiceStub(self.channel)
        self.scheduled_tasks = []
        self.daemon_task_receiver = threading.Thread(target=self.receive_tasks_results, daemon=True)

        self.daemon_task_receiver.start()

    def receive_tasks_results(self):
        while True:
            completed = []
            for task in self.scheduled_tasks:
                response = self.stub.GetTaskResult(task_service_pb2.TaskResultRequest(task_id=task.task_id))
                if response.result != "Task not completed yet":
                    completed.append(task)
                    task.setResult(response.result)
            for task in completed:
                self.scheduled_tasks.remove(task)
            # sleep for 1 second
            time.sleep(0.05)

    def wait_for_complete_all_tasks(self):
        completed = False
        while not completed:
            completed = True
            for t in self.scheduled_tasks:
                if t.result is None:
                    completed = False
                    break
            time.sleep(0.1)

    def run_task(self, task: ATask):
        self.scheduled_tasks.append(task)
        response = self.stub.RunTask(
            task_service_pb2.TaskRequest(task_id=task.task_id, task_name=task.task_name, args=task.args))
        print(response.status)


    def map_tasks(self, tasks: list[ATask]):
        """Запускает несколько задач и возвращает их результаты."""
        threads = []

        for tA in tasks:
            thread = threading.Thread(target=self.run_task, args=(tA,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()  # Ждем завершения всех потоков
        # wait for all tasks to complete
        completed = False

        while not completed:
            completed = True
            for t in tasks:
                if t.result is None:
                    completed = False
                    break
            time.sleep(0.5)

        return None



