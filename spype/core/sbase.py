"""
Contains base classes for spype.
"""

from copy import deepcopy

from spype.utils import context


# ---------------------------- Descriptors


class _OptionsDescriptor:
    """ descriptor for returning values of options """

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return context[self.name]


# -------------------- Spype base classes and descriptors


class _SpypeBase:
    """ base class that some allows using the MRO to access various spype Options """

    def copy(self):
        return deepcopy(self)


for item, value in context.items():
    setattr(_SpypeBase, item, _OptionsDescriptor(item))
