from SLM.appGlue.DesignPaterns.SingletonAtr import singleton


@singleton
class EventDispatcher:
    def __init__(self):
        self.listeners = {}
        self.events = {}
        self.event_handlers = {}

    def dispatch_event(self, event_name, *args, **kwargs):
        if event_name in self.events:
            for event_handler in self.events[event_name]:
                event_handler(*args, **kwargs)

    def register_listener(self, event_name, event_handler):
        """

        :param event_name:
        :param event_handler: eny callable object who can handle event
        :return:
        """
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(event_handler)
