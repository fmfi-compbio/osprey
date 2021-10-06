from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import os

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

mkl_root = os.getenv("MKLROOT", None)
if mkl_root is None:
    print("Please install MKL first and set MKLROOT")
    sys.exit()

ext_modules = [
    Pybind11Extension("osprey.osprey",
        ["src/caller.cpp"],
        cxx_std="17", 
        extra_compile_args=["-I" + os.path.join(mkl_root, "include"), "-march=native", "-DMKL_LP64", "-m64"],
        extra_link_args=[
            "-Wl,--start-group", 
            os.path.join(mkl_root, "lib", "intel64/libmkl_intel_lp64.a"), 
            os.path.join(mkl_root, "lib", "intel64/libmkl_sequential.a"),
            os.path.join(mkl_root, "lib", "intel64/libmkl_core.a"),
            "-Wl,--end-group",
            "-lpthread", "-lm", "-ldl"
        ]
    ),
]

install_requires = ["ont-fast5-api>=3.1.6", "numpy>=1.19", "scipy>=1.5", "python-dateutil>=2.8.2"]

setup(
    name="osprey",
    version=__version__,
    author="Vladimir Boza",
    author_email="bozavlado@gmail.com",
    url="https://github.com/fmfi-compbio/osprey",
    description="Fast ONT basecaller",
    long_description="",
    ext_modules=ext_modules,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
#    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=install_requires,
    scripts=["scripts/osprey_basecaller.py"],
    packages=["osprey"],
    package_data={'osprey': ['weights/net24dp.txt', 'weights/net24dp.txt.tabs']},
    include_package_data=True,
)
