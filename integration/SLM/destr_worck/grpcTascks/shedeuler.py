import socket
import datetime
import grpc
from concurrent import futures
import task_service_pb2
import task_service_pb2_grpc
from queue import Queue

class TaskScheduler(task_service_pb2_grpc.TaskServiceServicer):
    def __init__(self):
        self.tasks_queue = Queue()
        self.completed_tasks = {}
        self.averange_task_time = 0
        self.tasck_start_time = datetime.datetime.now()

    def RunTask(self, request, context):
        self.tasks_queue.put(request)
        return task_service_pb2.TaskResponse(status="Task queued")

    def GetTaskResult(self, request, context):
        if request.task_id in self.completed_tasks:
            return task_service_pb2.TaskResultRequest(task_id=request.task_id,result=self.completed_tasks[request.task_id])
        else:
            return task_service_pb2.TaskResultRequest(task_id=request.task_id,result="Task not completed yet")

    def GetTask(self, request, context):
        if self.tasks_queue.empty():
            return task_service_pb2.Task(task_id="", task_name="", args=[])
        task = self.tasks_queue.get()
        return task_service_pb2.Task(task_id=task.task_id, task_name=task.task_name, args=task.args)

    def TaskResult(self, request, context):
        print(f"Task {request.task_id} completed with result: {request.result}")
        self.completed_tasks[request.task_id] = request.result
        return task_service_pb2.Empty()

    def set_client_stub(self, client_stub):
        self.client_stub = client_stub  # Установка соединения с клиентом

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Подключение к фиктивному адресу, чтобы узнать IP-адрес сервера
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        s.close()
    return ip_address

print("Server IP Address:", get_local_ip())

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    task_service_pb2_grpc.add_TaskServiceServicer_to_server(TaskScheduler(), server)
    server.add_insecure_port('0.0.0.0:50051')
    print("server ip address:",get_local_ip())
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
