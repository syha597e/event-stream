import numpy as np


class Identity:
    def __call__(self, events):
        return events


class CropEvents:
    """Crops event stream to a specified number of events

    Parameters:
        num_events (int): number of events to keep
    """

    def __init__(self, num_events):
        self.num_events = num_events

    def __call__(self, events):
        if self.num_events >= len(events):
            return events
        else:
            start = np.random.randint(0, len(events) - self.num_events)
            return events[start:start + self.num_events]


class Jitter1D:
    """
    Apply random jitter to event coordinates
    Parameters:
        max_roll (int): maximum number of pixels to roll by
    """
    def __init__(self, sensor_size, var):
        self.sensor_size = sensor_size
        self.var = var

    def __call__(self, events):
        # roll x, y coordinates by a random amount
        shift = np.random.normal(0, self.var, len(events)).astype(np.int32)
        events['x'] += shift
        # remove events who got shifted out of the sensor size
        mask = (events['x'] >= 0) & (events['x'] < self.sensor_size[0])
        events = events[mask]
        return events


class Roll:
    """
    Roll event x, y coordinates by a random amount

    Parameters:
        max_roll (int): maximum number of pixels to roll by
    """
    def __init__(self, sensor_size, p, max_roll):
        self.sensor_size = sensor_size
        self.max_roll = max_roll
        self.p = p

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # roll x, y coordinates by a random amount
        roll_x = np.random.randint(-self.max_roll, self.max_roll)
        roll_y = np.random.randint(-self.max_roll, self.max_roll)
        events['x'] = np.add(events['x'],roll_x, out=events['x'], casting="unsafe")
        events['y'] = np.add(events['y'],roll_y, out=events['y'], casting="unsafe")
        # remove events who got shifted out of the sensor size
        mask = (events['x'] >= 0) & (events['x'] < self.sensor_size[0]) & (events['y'] >= 0) & (events['y'] < self.sensor_size[1])
        events = events[mask]
        return events


class Rotate:
    """
    Rotate event x, y coordinates by a random angle
    """
    def __init__(self, sensor_size, p, max_angle):
        self.p = p
        self.sensor_size = sensor_size
        self.max_angle = 2 * np.pi * max_angle / 360

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # rotate x, y coordinates by a random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        x = events['x'] - self.sensor_size[0] / 2
        y = events['y'] - self.sensor_size[1] / 2
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        events['x'] = (x_new + self.sensor_size[0] / 2).astype(np.int32)
        events['y'] = (y_new + self.sensor_size[1] / 2).astype(np.int32)
        # clip to original range
        events['x'] = np.clip(events['x'], 0, self.sensor_size[0])
        events['y'] = np.clip(events['y'], 0, self.sensor_size[1])
        return events


class Scale:
    """
    Scale event x, y coordinates by a random factor
    """
    def __init__(self, sensor_size, p, max_scale):
        assert max_scale >= 1
        self.p = p
        self.sensor_size = sensor_size
        self.max_scale = max_scale

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # scale x, y coordinates by a random factor
        scale = np.random.uniform(1/self.max_scale, self.max_scale)
        x = events['x'] - self.sensor_size[0] / 2
        y = events['y'] - self.sensor_size[1] / 2
        x_new = x * scale
        y_new = y * scale
        events['x'] = (x_new + self.sensor_size[0] / 2).astype(np.int32)
        events['y'] = (y_new + self.sensor_size[1] / 2).astype(np.int32)
        # remove events who got shifted out of the sensor size
        mask = (events['x'] >= 0) & (events['x'] < self.sensor_size[0]) & (events['y'] >= 0) & (events['y'] < self.sensor_size[1])
        events = events[mask]
        return events


class DropEventChunk:
    """
    Randomly drop a chunk of events
    """
    def __init__(self, p, max_drop_size):
        self.drop_prob = p
        self.max_drop_size = max_drop_size

    def __call__(self, events):
        max_drop_events = self.max_drop_size * len(events)
        if np.random.rand() < self.drop_prob:
            drop_size = np.random.randint(1, max_drop_events)
            start = np.random.randint(0, len(events) - drop_size)
            events = np.delete(events, slice(start, start + drop_size), axis=0)
        return events


def cut_mix_chunk(events, targets, max_num_events):
    """
    Cut and mix two event streams by a random event chunk. Input is a list of event streams.

    Args:
        events (dict): batch of event streams of shape (batch_size, num_events, 4)
        max_num_events (int): maximum number of events to mix
    """
    lengths = np.array([len(e) for e in events], dtype=np.int32)
    mix_events = np.random.randint(0, np.minimum(lengths, max_num_events))
    rand_index = np.random.permutation(len(events))

    # targets to mix
    targets_a = targets
    targets_b = targets[rand_index]
    lam = mix_events / lengths
    targets = targets_a * (1 - lam) + targets_b * lam

    # mix samples
    inds = np.random.randint(np.zeros_like(lengths), lengths - mix_events)
    cuts = [events[i, inds[i]:inds[i] + mix_events[i]] for i in range(len(events))]

    for i in range(len(events)):
        events[i, inds[i]:inds[i] + mix_events[i]] = cuts[rand_index[i]]

    return events, targets
