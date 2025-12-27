import asyncio
import threading
from typing import Optional, Callable

class Task:
    """Класс для представления задачи."""
    def __init__(self, name: str, coro, callback: Optional[Callable] = None):
        self.name = name
        self.coro = coro  # Асинхронная корутина
        self.callback = callback  # Колбэк, вызываемый при завершении задачи
        self.result = None
        self.exception = None
        self.status = "pending"  # Статус: pending, running, done, failed

    async def run(self, semaphore):
        """Выполняет задачу с использованием семафора и вызывает колбэк."""
        async with semaphore:  # Учитываем ограничение на количество задач
            try:
                self.status = "running"
                self.result = await self.coro
                self.status = "done"
            except Exception as e:
                self.exception = e
                self.status = "failed"
            finally:
                # Вызываем колбэк, если он задан
                if self.callback:
                    await self.callback(self)


class TaskDispatcher:
    """Класс-диспетчер для управления задачами с вачдогом."""
    def __init__(self, max_concurrent_tasks=3):
        self.tasks = []  # Очередь задач
        self.running_tasks = []  # Активные задачи
        self.max_concurrent_tasks = max_concurrent_tasks  # Максимальное количество одновременных задач
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.watchdog_active = False  # Состояние вачдога
        self.watchdog_task = None  # Фоновая задача вачдога

    def add_task(self, name: str, coro, callback: Optional[Callable] = None):
        """Добавляет задачу в очередь диспетчера."""
        task = Task(name, coro, callback)
        self.tasks.append(task)

    async def process_tasks(self):
        """Обрабатывает задачи из очереди до тех пор, пока они есть."""
        while self.watchdog_active:
            if self.tasks:  # Если есть задачи в очереди
                task = self.tasks.pop(0)  # Забираем первую задачу
                self.running_tasks.append(asyncio.create_task(task.run(self.semaphore)))
            else:
                await asyncio.sleep(0.1)  # Ждём появления новых задач

    def start_watchdog(self):
        """Запускает вачдог."""
        if not self.watchdog_active:
            self.watchdog_active = True
            self.watchdog_task = asyncio.create_task(self.process_tasks())
            print("Вачдог запущен")

    async def stop_watchdog(self):
        """Останавливает вачдог."""
        self.watchdog_active = False
        if self.watchdog_task:
            await self.watchdog_task
            self.watchdog_task = None
            print("Вачдог остановлен")

    async def wait_for_completion(self):
        """Ждёт завершения всех активных задач."""
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks)
            self.running_tasks.clear()

    def get_status(self):
        """Возвращает статус задач в очереди и активных задач."""
        return [
            {
                "name": task.name,
                "status": task.status,
                "result": task.result,
                "exception": str(task.exception) if task.exception else None,
            }
            for task in self.tasks + self.running_tasks
        ]


# Пример использования
async def example_task(name, delay):
    print(f"Задача {name} началась")
    await asyncio.sleep(delay)
    print(f"Задача {name} завершилась")
    return f"Результат {name}"

async def task_callback(task):
    """Колбэк, вызываемый при завершении задачи."""
    if task.status == "done":
        print(f"Колбэк: {task.name} успешно завершена с результатом: {task.result}")
    elif task.status == "failed":
        print(f"Колбэк: {task.name} завершилась с ошибкой: {task.exception}")

# Предположим, TaskDispatcher уже определён
async def run_dispatcher(dispatcher):
    dispatcher.start_watchdog()
    await asyncio.sleep(5)  # Дождаться обработки задач в течение 5 секунд
    await dispatcher.stop_watchdog()

# Запуск диспетчера в отдельном потоке
def start_dispatcher_in_thread(dispatcher):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_dispatcher(dispatcher))
    loop.close()

def main_sync():
    dispatcher = TaskDispatcher(max_concurrent_tasks=2)

    # Добавляем задачи
    dispatcher.add_task("Task 1", example_task("Task 1", 2))
    dispatcher.add_task("Task 2", example_task("Task 2", 1))
    dispatcher.add_task("Task 3", example_task("Task 3", 3))

    # Запуск диспетчера в бесконечном цикле через поток
    thread = threading.Thread(target=start_dispatcher_in_thread, args=(dispatcher,))
    thread.start()

    # Добавляем задачи в реальном времени
    import time
    time.sleep(3)  # Симуляция работы синхронного кода
    dispatcher.add_task("Task 3", example_task("Task 3", 2), callback=task_callback)

    # Останавливаем диспетчер через 10 секунд
    time.sleep(10)
    asyncio.run(dispatcher.stop_watchdog())
    thread.join()


if __name__ == "__main__":
    main_sync()
