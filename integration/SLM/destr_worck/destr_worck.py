import copy
import json

from time import sleep, time
import xmlrpc.client
import os
import threading
import uuid
from xmlrpc.server import SimpleXMLRPCServer

from sqlmodel import SQLModel, create_engine, Field, Session, select, col


class task:
    def __init__(self, action: str, args, kwargs):
        self.id = str(uuid.uuid4())
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.status = "pending"
        self.result = None
        self.log = []
        self.need_callback = True
        self.token = "default"

    def wait_result(self, worc_manager_inst):
        while self.status == "pending":
            sleep(0.1)
            state = worc_manager_inst.broker.get_task_state(self.id)
            if state is not None:
                if state.status == "completed":
                    self.status = "completed"
                    self.result = state.result
                if state.status == "error":
                    self.status = "error"
                    self.log.append(state.log)
        return self.result


class SQLModel_message(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    action_id: str
    action: str
    need_callback: bool
    args: str
    kwargs: str
    status: str
    result: str
    log: str
    token: str = Field(default="default")
    dependency: str = Field(default="")
    execution_token: str = Field(default="default")


class SQLModel_action_callback(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    action_id: str
    token: str


class SqlModel_action_result(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    action_id: str
    action: str
    args: str
    kwargs: str
    status: str
    result: str
    log: str
    token: str = Field(default="default")


class SQLModel_SQLLite_broker:

    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path + "/message_broker"):
            os.makedirs(self.path + "/message_broker")
        sql_str = f'sqlite:///{self.path}/message_broker/message_broker.db'
        self.engine = create_engine(sql_str, echo=False)
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def get_new_session(self):
        return Session(self.engine)

    def push_task(self, task_inst, commit=True):
        args_json = json.dumps(task_inst.args)
        kwargs_json = json.dumps(task_inst.kwargs)
        result_json = json.dumps(task_inst.result)
        log_json = json.dumps(task_inst.log)
        task = SQLModel_message(
            action_id=task_inst.id,
            action=task_inst.action,
            need_callback=task_inst.need_callback,
            args=args_json,
            kwargs=kwargs_json,
            status=task_inst.status,
            result=result_json,
            log=log_json,
            token=task_inst.name
        )
        self.session.add(task)
        if commit:
            self.session.commit()

    def commit(self):
        self.session.commit()

    def pop_task(self, compactible_tascks: list):
        query = (select(SQLModel_message).where(SQLModel_message.status == "pending").
                 where(col(SQLModel_message.action).in_(compactible_tascks)).limit(1))
        result = self.session.exec(query).one_or_none()
        self.session.delete(result)
        self.session.commit()
        if result is not None:
            task_inst = task(result.action, json.loads(result.args), json.loads(result.kwargs))
            task_inst.id = result.action_id
            task.status = result.status
            task.result = json.loads(result.result)
            task.log = json.loads(result.log)
            task.token = result.name
            task.need_callback = result.need_callback
            return task_inst
        return None

    def store_action_result(self, task_inst):
        args_json = json.dumps(task_inst.args)
        kwargs_json = json.dumps(task_inst.kwargs)
        result_json = json.dumps(task_inst.result)
        log_json = json.dumps(task_inst.log)
        task_res = SqlModel_action_result(
            action_id=task_inst.id,
            action=task_inst.action,
            args=args_json,
            kwargs=kwargs_json,
            status=task_inst.status,
            result=result_json,
            log=log_json,
            token=task_inst.name
        )
        callback_item = SQLModel_action_callback(
            action_id=task_inst.id,
            token=task_inst.name
        )
        self.session.add(callback_item)
        self.session.add(task_res)
        self.session.commit()

    def delete_action_result(self, task_inst):
        query = select(SqlModel_action_result).where(SqlModel_action_result.action_id == task_inst.id)
        result = self.session.exec(query).one_or_none()
        if result is not None:
            self.session.delete(result)
            self.session.commit()

    def on_task_completed(self, task_inst):
        if task_inst.need_callback:
            self.store_action_result(task_inst)

    def get_completed(self, token="default"):
        query = select(SQLModel_action_callback).where(SQLModel_action_callback.token == token)
        result = self.session.exec(query).all()
        return result

    def get_task_state(self, id):
        query = select(SQLModel_action_callback).where(SQLModel_action_callback.action_id == id)
        result = self.session.exec(query).one_or_none()
        return result

    def count_pending(self):
        query = select(SQLModel_message).where(SQLModel_message.status == "pending")
        result = self.session.exec(query).all()
        return len(result)

    def count_completed(self):
        query = select(SQLModel_action_callback)
        result = self.session.exec(query).all()
        return len(result)


class TaskRepository:
    actions = {}

    @staticmethod
    def register_task(funct):
        TaskRepository.actions[funct.__name__] = funct
        return funct


class worc_manager:
    def __init__(self, url="http://localhost:8000"):
        self.proxy = xmlrpc.client.ServerProxy(url + "/RPC2",allow_none=True,verbose=False)

    def create_new_task(self, action: str,*args, **kwargs ):
        task_inst = task(action, args, kwargs)
        return task_inst

    def delay(self,task):
        self.proxy.register_task(task)





class work_planer:
    def __init__(self):  # , worc_manager_ins: worc_manager):
        # self.worc_manager_ins: worc_manager = worc_manager_ins
        self.workers_pool = []
        self.broker = SQLModel_SQLLite_broker(r"D:\My Drive\cool_tools\destr_worck")
        self.end = False
        self.task_count = 0
        self.completed_task_count = 0
        self.main_thread_call_list = []
        self.scheduled_task_list = []
        self.execution_token = "default"
        self.rpc_server = SimpleXMLRPCServer(("localhost", 8000), allow_none=True)
        # register some functions
        self.rpc_server.register_function(self.register_task, "register_task")
        self.server_thread = threading.Thread(target=self.rpc_server.serve_forever)
        self.refresh_info_interval:float = 5
        self.refresh_time = 0
        self.start_time = 0
        self.average_task_time = 0


    def register_task(self, task_dict):
        task_inst = task(task_dict["action"], task_dict["args"], task_dict["kwargs"])
        task_inst.id = task_dict["id"]
        task_inst.need_callback = task_dict["need_callback"]
        self.scheduled_task_list.append(task_inst)

    def commit_tasks(self):
        if len(self.scheduled_task_list) == 0:
            return
        for task in self.scheduled_task_list:
            self.broker.push_task(task, commit=False)
        self.broker.commit()
        self.scheduled_task_list = []

    def call_on_main_thread(self, function, *args, **kwargs):
        self.main_thread_call_list.append((function, args, kwargs))

    def daemon(self):
        self.start_time = time()
        self.server_thread.start()
        while not self.end:
            curent_time = time()
            diference = curent_time - self.start_time
            if diference > self.refresh_info_interval+self.refresh_time:
                self.start_time = curent_time
                start_time = time()
                self.task_count = self.broker.count_pending()
                self.completed_task_count = self.broker.count_completed()
                print(f"task count: {self.task_count} completed tasck count: {self.completed_task_count}")
                print(f"average task time: {self.average_task_time}")
                dif_info_time = time() - start_time
                self.refresh_time = dif_info_time

            self.commit_tasks()

            for worker in self.workers_pool:
                if worker.worker_state == "idle":
                    worker.current_task = self.pop_task(worker.get_compatible_task_list())
                    if worker.current_task is not None:
                        self.average_task_time = (self.average_task_time + worker.exec_time) / 2
                        worker.start()

            for call in self.main_thread_call_list:
                function, args, kwargs = call
                function(*args, **kwargs)
            self.main_thread_call_list = []
            sleep(0.05)
            if self.task_count == 0:
                sleep(0.5)

    def add_worker(self, count):
        for i in range(count):
            self.workers_pool.append(worker(self))

    def start(self):
        self.daemon()
        for worker in self.workers_pool:
            worker.worker_thread.join()

    def stop(self):
        self.end = True

    def pop_task(self, compactible_tascks: list):
        try:
            task_inst = self.broker.pop_task(compactible_tascks)
        except Exception as e:
            return None
        return task_inst

    def task_completed(self, task_inst):
        self.call_on_main_thread(self.broker.on_task_completed, task_inst)


class worker:
    def __init__(self, work_planer_inst):
        self.worker_state = "idle"
        self.work_planer = work_planer_inst
        self.current_task: task | None = None
        self.worker_thread = threading.Thread(target=self.do_task)
        self.worker_message = ""
        self.exec_time = 0

    def get_compatible_task_list(self):
        return TaskRepository.actions.keys()

    def start(self):
        self.worker_thread = threading.Thread(target=self.do_task)
        self.worker_thread.start()

    def execute_task(self):
        start_time = time()
        kwargs = copy.copy(self.current_task.kwargs)
        kwargs["worker"] = self
        try:
            result = self.execute_task_function(self.current_task.action, self.current_task.args, kwargs)
            self.current_task.result = result
            self.exec_time = time() - start_time
            self.current_task.status = "completed"
        except Exception as e:
            self.current_task.log.append(str(e))
            self.current_task.result = None
            self.exec_time = time() - start_time
            self.current_task.status = "error"

    def task_completed(self):
        self.work_planer.task_completed(self.current_task)
        self.current_task = None

    def execute_task_function(self, action, args, kwargs):
        return TaskRepository.actions[action](*args, **kwargs)

    def do_task(self):

        self.worker_state = "working"
        self.execute_task()
        self.task_completed()
        self.worker_state = "idle"


class RPC_worker(worker):
    def __init__(self, work_planer_inst, uri="http://localhost:8000"):
        super().__init__(work_planer_inst)
        self.uri = uri

    def get_proxy(self):
        import xmlrpc.client
        return xmlrpc.client.ServerProxy(self.uri + "/RPC2")

    def get_compatible_task_list(self):
        return self.get_proxy().get_compatible_task_list()

    def execute_task_function(self, action, *args, **kwargs):
        return self.get_proxy().do_task(action, *args, **kwargs)


class result_client:
    def __init__(self, worc_manager_inst):
        self.worc_manager: worc_manager = worc_manager_inst
        self.on_task_completed_delegate = None
        self.list_of_completed_tasks_id = []

    def on_task_completed(self, task_inst):
        if self.on_task_completed_delegate is not None:
            self.on_task_completed_delegate(task_inst)


