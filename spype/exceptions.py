"""
Custom Exceptions for Sflow (these are to be used very sparingly!
"""


class SpypeException(Exception):
    """
    Base exception class for spype
    """


# --------- Exceptions for controlling flow of Task Queue


class UnresolvedDependency(SpypeException):
    """
    Exception raised when a task needs to be left appended to the task queue.

    This can happen if the task is a future that has yet to return, or if the
    task uses a dependency that has not yet been resolved.
    """


class TaskReturnedNone(SpypeException):
    """
    Exception that is raised when a task  returns None.

    This happens when a filter condition fails, for example
    """


class ReturnCallbackValue(SpypeException):
    """
    Exception raised during task execution, usually by a callback or
    surrounding control structure, that indicates all work on a task should
    stop.
    """


class ExitTask(SpypeException):
    """
    Exception for stopping task execution and returning None
    """


# ----------- Exceptions for validating pypes


class InvalidPype(SpypeException):
    """
    Base class for exceptions that are raised with invalid digraphs.
    """


class UnresolvableNetwork(InvalidPype):
    """
    Exception raised when an invalid digraph is detected.

    This can happen if the digraph has unconditional cycles or
    has circular dependencies.
    """


class IncompatibleTasks(InvalidPype, TypeError):
    """
    Exception that is raised when incompatible tasks are hooked together.
    """


# ------------ Type Exceptions


class SpypeTypeException(SpypeException, TypeError):
    """
    Base exception for Spype's type checking stuff.
    """


class NoReturnAnnotation(SpypeTypeException):
    """
    Raised when a return annotation on a signature is not found.
    """
