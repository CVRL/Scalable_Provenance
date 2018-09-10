def walk(data, pre=None, post=None):
    """Calls pre and post functions for each node in a tree of data."""
    if pre:
        pre(data)
    if isinstance(data, dict):
        for value in data.values():
            walk(value, pre, post)
    elif isinstance(data, list):
        for value in data:
            walk(value, pre, post)
    if post:
        post(data)


def transform(data, pre=None, post=None):
    """Transforms each node in a tree of data."""
    if pre:
        data = pre(data)
    if isinstance(data, dict):
        data = {key: transform(value, pre, post) for key, value in data.items()}
    elif isinstance(data, list):
        data = [transform(value, pre, post) for value in data]
    if post:
        data = post(data)
    return data
