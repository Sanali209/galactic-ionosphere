import time
from threading import Thread

from loguru import logger

from SLM.appGlue.core import Service, Allocator



class BGTaskState:
    """
    The class that represents the state of a background task.
    Can be used to track the progress of a task.
    Possible states are:
    - in_progress: The task is in progress.
    - done: The task is done.
    - error: The task has encountered an error.
    - cancel: The task has been canceled.
    """

    def __init__(self):
        self.status = "in_progress"
        self.progress = 0
        self.progress_message = ""
        self.result = None


class BGTask:
    """
    The class that represents a background task.

    """

    def __init__(self, *args, **kwargs):
        self.manager: BGWorker = None
        self.name = None
        self.cancel_names = []
        """ cancel tasks by names on time of add in queue"""
        self.exclude_names = []
        """prevent add task if name in exclude exists in queue"""

        self.args = args
        self.kwargs = kwargs
        self.done_callback = None
        self.state: BGTaskState = BGTaskState()
        self.queue_name = 'default'
        self.generator = self.task_function()

    def report_progress(self):
        logger.info(
            f"Tascks:{len(self.manager.tasks_queues[self.queue_name].tasks)} Task {self.name} progress: {self.state.progress}")

    def on_done(self, callback):
        self.done_callback = callback

    def fire_done(self):
        self.report_progress()
        if self.done_callback:
            self.done_callback(self)

    def set_done(self, result=None):
        self.state.status = "done"
        self.state.result = result
        self.fire_done()

    def set_error(self, error):
        self.state.status = "error"
        self.state.result = error
        self.fire_done()

    def set_cancel(self):
        self.state.status = "cancel"
        self.state.result = "cancel"
        self.fire_done()

    def set_work_function(self, func):
        self.generator = func()

    def task_function(self):
        """
        The function that performs the task. This function should be overridden in subclasses.
        :return:  one 'yield' for one step of task or 'return' for error generation
        """
        print("work_gen")
        yield

    def run(self):
        if self.state.status == "done" or self.state.status == "cancel" or self.state.status == "error":
            return
        try:
            generator_resalts = next(self.generator)
            self.report_progress()
            if generator_resalts == "done":
                self.set_done(generator_resalts)
        except StopIteration as e:
            self.set_done()
        except Exception as e:
            logger.error(f"Error in task {self.name}: {e}")
            self.set_error(e)



class TaskQueueSeq:
    def __init__(self, name='default'):
        self.parent_task_manager = None
        self.tasks = []
        self.name = name
        self.cur_task = None
        self.generator = self.do_tasks_gen()

    def get_tasks_names_in_queue(self) -> list[str]:
        list_of_tasks_names = [task.name for task in self.tasks]
        if self.cur_task is not None:
            list_of_tasks_names.append(self.cur_task.name)
        return list_of_tasks_names

    def get_progress_mean(self):
        if len(self.tasks) == 0:
            return 0
        return sum([task.state.progress for task in self.tasks]) / len(self.tasks)

    def is_empty(self):
        return len(self.tasks) == 0

    def add_task(self, task: BGTask, ignore_excludes=False):
        task.queue_name = self.name
        # todo write task to disk cache

        if not ignore_excludes:
            for name in task.exclude_names:
                task_names = self.get_tasks_names_in_queue()  # check if task in exclude
                if name in task_names:
                    return
        self.tasks.append(task)
        self.update_generator()

    def add_first(self, task: BGTask, ignore_excludes=False):
        task.queue_name = self.name
        # todo write task to disk cache

        if not ignore_excludes:
            for name in task.exclude_names:
                task_names = self.get_tasks_names_in_queue()
                if name in task_names:
                    return
        self.tasks.insert(0, task)
        self.update_generator()

    def remove_task(self, task: BGTask):
        if task in self.tasks:
            self.tasks.remove(task)
            self.update_generator()

    def update_generator(self):
        self.generator = self.do_tasks_gen()

    def do_tasks(self):
        try:
            next(self.generator)
        except StopIteration as e:
            return

    def do_tasks_gen(self):
        while not self.is_empty():
            yield self.do_next_task()

    def do_next_task(self):
        self.cur_task = self.tasks.pop(0)
        if (self.cur_task.state.status == "done" or
                self.cur_task.state.status == "cancel" or
                self.cur_task.state.status == "error"):
            pass
        else:
            self.cur_task.run()
            if self.cur_task.state.status != "done":
                self.tasks.append(self.cur_task)
        self.cur_task = None


dask_en = False
if dask_en:
    # todo: integrate dask
    from dask.distributed import Client

    dusk_client = Client(processes=False)


    class BGDuskTusk(BGTask):
        def __init__(self):
            super().__init__()
            # dushbord on http://localhost:8787/

            self.dusk_funct = self.work_func

        def set_work_function(self, func):
            self.dusk_funct = func

        def task_function(self):
            future = dusk_client.submit(self.dusk_funct)
            if future.done():
                result = future.result()
                self.set_done(result)
            else:
                yield

        def work_func(self):
            print(self)


class BGCancelTask(BGTask):
    def __init__(self):
        super().__init__()
        self.exclude = []

    def run(self):
        for task in self.manager.tasks_queues[self.queue_name].tasks:
            if task.name in self.cancel_names and task not in self.exclude:
                task.set_cancel()
        self.set_done()


class BGWorker(Service):
    """
    The class that manages the background tasks. Runs in a separate thread.
    Run execution tread on creation of instance of class.
    """

    def __init__(self):
        super().__init__()
        self.tasks_queues = {'default': TaskQueueSeq()}
        self.run_t = True
        # todo: watch on worck state
        self.work_thread: Thread = Thread(target=self.tusks_execute)
        self.work_thread.start()

    def add_queue(self, name, type_: type):
        self.tasks_queues[name] = type_(name)

    def tusks_execute(self):
        try:
            while self.run_t:
                if all([queue.is_empty() for queue in [*self.tasks_queues.values()]]):
                    time.sleep(1)
                for queue in [*self.tasks_queues.values()]:
                    queue.do_tasks()
        finally:
            logger.info("BGWorker: stop")
            self._finalize()

    def _finalize(self):
        self.run_t = False
        self.work_thread.join()

    def stop(self):
        self.run_t = False

    def add_task(self, task: BGTask, ignore_excludes=False):
        task.manager = self
        self.tasks_queues[task.queue_name].add_task(task, ignore_excludes)

    def add_task_first(self, task: BGTask, ignore_excludes=False):
        task.manager = self
        self.tasks_queues[task.queue_name].add_first(task, ignore_excludes)

    def cancel_task_by_names(self, names: list, queue_name='default'):
        task = BGCancelTask()
        task.queue_name = queue_name
        task.cancel_names = names
        self.add_task(task)


Allocator.res.register(BGWorker())
