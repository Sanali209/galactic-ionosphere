import json

import grpc
import time

import torch
from transformers import pipeline

import task_service_pb2
import task_service_pb2_grpc
from concurrent.futures import ThreadPoolExecutor

from SLM.vision.imagetotext.ImageToLabel import multiclass_comix_bf

# Check if a GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1
pipline_name = "sanali209/sketch_filter"
pipeline1=pipeline("image-classification", model=pipline_name, framework="pt",device=device)

class TaskFactory:
    """Фабрика задач, выбирает алгоритм выполнения в зависимости от имени задачи."""

    @staticmethod
    def execute_task(task_name, args):
        tasks = {
            "vision_comics_bw": TaskFactory.vision_comics_bw,
            "task2": TaskFactory.task2,
            # Добавьте новые задачи сюда
        }

        if task_name in tasks:
            return tasks[task_name](args)
        else:
            return "Unknown task"

    @staticmethod
    def vision_comics_bw(args):
        resultsl = []
        for arg in args:
            image = arg
            results = pipeline1(image)
            print(f"Task vision_comics_bw processing")
            # Пример выполнения задачи 1
            # save result as json string
            resultsl.append(results)
        results = json.dumps(resultsl)
        return results

    @staticmethod
    def task2(args):
        # Пример выполнения задачи 2
        return f"Task 2 processed with args: {args}"


class Worker:
    def __init__(self,connection_str='localhost:50051'):
        self.channel = grpc.insecure_channel(connection_str)
        self.stub = task_service_pb2_grpc.TaskServiceStub(self.channel)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def fetch_task(self):
        print("Worker started")
        while True:
            task = self.stub.GetTask(task_service_pb2.Empty())
            if task.task_id:
                print(f"Received task: {task.task_id}, name: {task.task_name}")
                #self.executor.submit(self.process_task, task)
                self.process_task(task)
            time.sleep(0.01)

    def process_task(self, task):
        result = TaskFactory.execute_task(task.task_name, task.args)
        self.stub.TaskResult(task_service_pb2.TaskResultRequest(task_id=task.task_id, result=result))


if __name__ == '__main__':
    worker = Worker(connection_str="5.tcp.eu.ngrok.io:19074")
    worker.fetch_task()
