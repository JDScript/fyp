import importlib


def retrieve_class_from_string(target: str):
    module, cls = target.rsplit(".", 1)
    module = importlib.import_module(module)
    return getattr(module, cls)
