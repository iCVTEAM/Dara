def _init():
    global _global_dict
    _global_dict = {}
def set_value(key, value):
    _global_dict[key] = value
def get_value(key):
        return _global_dict[key]