# Observer pattern classes.

class Observable:
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for obs in self._observers:
            obs.notify(self, *args, **kwargs)


class Observer:
    def __init__(self):
        pass

    def notify(self, observable, *args, **kwargs):
        print("Got", args, kwargs, "From", observable)
