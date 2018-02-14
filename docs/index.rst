spype
=====

Spype is a [s]imple [py]thon [p]ipelin[e] library.

It facilitates the creation of lightweight, expressive data pipelines.

Spype has three main design goals:

* Independent

    Spype has no required dependencies. It does, however, require `graphviz <https://graphviz.readthedocs.io/en/stable/>`_. for visualization and `pytest <https://docs.pytest.org/en/latest/>`_ for running the test suite.


* Simple, Declarative API

    Spype provides an intuitive, declarative API for hooking together python callables in order to create arbitrarily complex data pipelines.


* Disciplined

    Spype can (optionally) provide runtime type-checking and compatibility validation in order to help you find bugs faster, and give you more confidence in your data pipelines.

Liscence: BSD

WARNING: spype is brand new and experimental, dont use it in production until it matures a little, and expect frequent API changes.

Documentation:

.. toctree::
   :maxdepth: 1

   notebooks/intro.ipynb

   notebooks/should_i_use.ipynb

.. toctree::
   :maxdepth: 2

   notebooks/tutorial.ipynb

   contributing.rst


API
===
.. toctree::
   :maxdepth: 1

   tasks.rst

   wraps.rst

   pypes.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
