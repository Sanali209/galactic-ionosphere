import threading
import time

from SLM.appGlue.core import Service, Allocator

from loguru import logger


class TimerManager(Service):
    def __init__(self):
        super().__init__()
        self.timers = []
        self.timer_thread = threading.Thread(target=self.run, daemon=True)
        self._stop_event = threading.Event()
        self.sleep_time = 0.2
        """ The time to sleep between timer checks. """
        self.verbose = False

    def init(self, config):
        self.timer_thread.start()

    def run(self):
        try:
            while not self._stop_event.is_set():
                for timer_instance in self.timers:
                    timer_instance.check()
                time.sleep(self.sleep_time)
        except KeyboardInterrupt:
            logger.warning("Timer thread interrupted by user.")
        finally:
            self._finalize()

    def stop(self):
        self._stop_event.set()

    def _finalize(self):
        self._stop_event.set()
        try:
            self.timer_thread.join()
        except Exception as e:
            logger.error(f"Error stopping timer thread: {e}")

        for timer_instance in self.timers:
            timer_instance.destroy()
        self.timers = []


class Timer:
    """
    A class representing a timer that runs in a separate thread.
    It emits events with the elapsed time at regular intervals.
    """

    def __init__(self, interval):
        super().__init__()
        TimerManager.instance().timers.append(self)
        self.elapsed_time = 0  # Total elapsed time since start
        self.last_time = time.time()  # Time of the last event
        self.running = False  # Flag to indicate if the timer is running
        self.observers = []  # List of observer objects to notify
        self.event_time = interval  # Time at which the timer event will be triggered
        self.single = False
        self.name = "Timer"
        """name of the timer"""
        self.owner = None

    def start(self):
        """
        Starts the timer thread.
        """
        self.elapsed_time = 0
        if not self.running:
            self.last_time = time.time()
        self.running = True

    def destroy(self):
        TimerManager.instance().timers.remove(self)

    def register(self, observer):
        """
        Registers an observer object that will be notified of timer events.
        """
        self.observers.append(observer)

    def check(self):
        """
        Checks if the timer has elapsed and notifies the observers.
        """
        if self.running:
            current_time = time.time()
            self.elapsed_time += current_time - self.last_time
            last_from_prev = current_time - self.last_time
            self.last_time = current_time
            for observer in self.observers:
                try:
                    observer.on_timer_event(self, self.elapsed_time, last_from_prev)
                except Exception as e:
                    if TimerManager.instance().verbose:
                        print(f"Error in observer: {e}")
            if self.elapsed_time >= self.event_time:
                for observer in self.observers:
                    try:
                        observer.on_timer_notify(self)
                    except Exception as e:
                        if TimerManager.instance().verbose:
                            print(f"Error in observer: {e}")
                if self.single:
                    self.stop()
                self.reset()

    def stop(self):
        """
        Stops the timer thread.
        """
        self.running = False

    def reset(self):
        """
        Resets the elapsed time to zero.
        """
        self.elapsed_time = 0


class TimerBuilder:
    def __init__(self):
        self.timer = Timer(0)
        self.observer = TObserver()

    def set_interval(self, interval):
        self.timer.event_time = interval
        return self

    def set_single(self, single):
        self.timer.single = single
        return self

    def set_name(self, name):
        self.timer.name = name
        return self

    def set_on_timer_notyfy(self, on_timer_notify):
        self.observer.on_timer_notify = on_timer_notify
        return self

    def build(self):
        self.timer.register(self.observer)
        self.timer.start()
        return self.timer


class TObserver:
    """
    todo: refactor to callable
    An abstract class representing an observer of timer events.
    """

    def on_timer_event(self, elapsed_time, last_from_prev):
        """
        This method is called by the timer thread whenever a new elapsed time is available.
        Subclasses should implement this method to handle the event.
        """
        raise NotImplementedError

    def on_timer_notify(self):
        """
        This method is called by the timer thread whenever a new elapsed time is available.
        Subclasses should implement this method to handle the event.
        """
        print("Timer event!")


if __name__ == "__main__":
    class TestObserver(TObserver):
        def on_timer_event(self, elapsed_time, last_from_prev):
            print(f"Elapsed time: {elapsed_time}" + f" last from prev: {last_from_prev}")

        def on_timer_notify(self):
            print("Timer event!")


    timer = Timer(5)
    timer.name = "Test Timer"
    observer = TestObserver()
    timer.register(observer)
    timer.start()
