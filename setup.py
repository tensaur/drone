# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "simulator/cy_env.pyx", compiler_directives={"language_level": "3"}
    ),
    zip_safe=False,
    py_modules=["simulator"],
    include_dirs=[numpy.get_include()],
)
