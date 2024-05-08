import os

import pybind11
from setuptools import Extension, setup

functions_module = Extension(
    name="EntropyCodec",
    sources=["wrapper.cpp"],
    include_dirs=[
        os.path.join(pybind11.__path__[0], "include"),
    ],
)

setup(ext_modules=[functions_module], options={"build_ext": {"build_lib": ".."}})
