"""
Utility functions.
"""

import collections

def singlequote(x):
    """Return x wraped in single quotes."""

    return "'{}'".format(x)

def isfloat(x):
    """Is x castable to a float?"""

    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return True

def tofloat(x, default = 0):
    """Return x cast to a float."""

    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)

def flatten(iterable):
    """Return a flattened version of an iterable."""

    for x in iterable:
        if hasattr(x, "__iter__"):
            yield from flatten(x)
        else:
            yield x

def join(iterables, key = lambda x: x, select = min, default = None):
    """Join iterables on a given key.

    Parameters:
    iterables -- iterables to join
    key -- function to return key from value
    select -- function to select next key from a set of keys
    default -- default value to use when key not present
    """

    exhausted = object()
    iterators = [iter(i) for i in iterables]
    values = [next(i, exhausted) for i in iterators]
    while any([v is not exhausted for v in values]):
        keys = [key(v) if v is not exhausted else exhausted for v in values]
        selected = select([k for k in keys if k is not exhausted])
        yield [v if k == selected else default
            for k, v in zip(keys, values)]
        values = [next(i, exhausted) if k == selected else v
            for k, v, i in zip(keys, values, iterators)]

def slidingwindow(iterable, size, stride = 1):
    """Generate windows over an iterable.
    
    Parameters:
    iterable -- iterable to window
    size -- size of window
    stride -- stride of window
    """

    iterator = iter(iterable)
    window = collections.deque([], size)
    try:
        while len(window) < size:
            window.append(next(iterator))
        while True:
            yield list(window)
            for i in range(stride):
                window.append(next(iterator))
    except StopIteration:
        pass

def batches(iterable, size, partial = False):
    """Generate batches from an iterable.

    Parameters:
    iterable -- iterable to batch
    size -- size of batches
    partial -- return a short batch when iterable exhausted
    """

    batch = []
    for value in iterable:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch and partial:
        yield batch
