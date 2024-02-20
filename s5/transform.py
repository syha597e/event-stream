import numpy as np


class CropEvents:
    """Crops event stream to a specified number of events

    Parameters:
        num_events (int): number of events to keep
    """

    def __init__(self, num_events):
        self.num_events = num_events

    def __call__(self, events):
        if self.num_events > len(events):
            return events
        else:
            start = np.random.randint(0, len(events) - self.num_events)
            return events[start:start + self.num_events]