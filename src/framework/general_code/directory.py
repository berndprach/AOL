
from typing import Dict


class Directory:
    def __init__(self):
        self.items: Dict[frozenset, any] = {}

    def __call__(self, *args):
        return self.get(*args)

    def __getitem__(self, item):  # just to make it more compatible with factory!
        return self.get(*item)

    def __str__(self):
        return "Directory with the following keys:" + "".join([f"\n   {key}" for key in self.keys])

    @property
    def keys(self):
        return self.items.keys()

    def register(self, product: callable, *args):
        key = frozenset(args)
        self.items[key] = product

    def get(self, *args):
        key = frozenset(args)
        if key in self.items:
            return self.items[key]
        else:
            raise ValueError(f"Directory entry with key {args} not defined yet! ({self})")

    def as_dict(self):
        return self.items

    def create(self, product_name, *args, **kwargs):
        return self(product_name, *args)(**kwargs)

