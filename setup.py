#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from spype.version import __version__
import glob
from os.path import join, abspath, dirname, isdir, exists
from setuptools import setup

here = dirname(abspath(__file__))

with open(join(here, "spype", "version.py"), "r") as fi:
    content = fi.read().split("=")[-1].strip()
    __version__ = content.replace('"', "").replace("'", "")

with open("README.rst") as readme_file:
    readme = readme_file.read()


# --- get sub-packages
def find_packages(base_dir="."):
    """ setuptools.find_packages wasn't working so I rolled this """
    out = []
    for fi in glob.iglob(join(base_dir, "**", "*"), recursive=True):
        if isdir(fi) and exists(join(fi, "__init__.py")):
            out.append(fi)
    out.append(base_dir)
    return out


requirements = []

extra_require = dict(
    plot=["graphviz"], docs=["sphinx", "sphinx-autodoc-typehints", "nbsphinx"]
)

test_requirements = ["pytest", "pytest-runner"]

setup_requirements = []

setup(
    name="spype",
    version=__version__,
    description="A lightweight data pipeline library",
    long_description=readme,
    author="Derrick Chambers",
    author_email="djachambeador@gmail.com",
    url="https://github.com/d-chambers/spype",
    packages=find_packages("spype"),
    package_dir={"spype": "spype"},
    include_package_data=True,
    install_requires=requirements,
    extra_require=extra_require,
    license="BSD",
    zip_safe=False,
    keywords="data pipeline",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    python_requires=">=3.6",
)
