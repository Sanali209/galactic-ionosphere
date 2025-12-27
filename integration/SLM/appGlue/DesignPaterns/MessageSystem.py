


class MessageSystem:
    subscribers = {}

    @staticmethod
    def Subscribe( message, subscriber, callback):
        MessageSystem.subscribers.setdefault(message, []).append((subscriber, callback))

    @staticmethod
    def Unsubscribe( message, subscriber):
        if message in MessageSystem.subscribers:
            MessageSystem.subscribers[message] = [
                sub for sub in MessageSystem.subscribers[message] if sub[0] != subscriber
            ]
    @staticmethod
    def SendMessage( message, *args, **kwargs):
        if message in MessageSystem.subscribers:
            for subscriber, callback in MessageSystem.subscribers[message]:
                callback(*args, **kwargs)





