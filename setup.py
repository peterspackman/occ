import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

setup(
    name="occ",
    version="0.0.1",
    description="a minimal example package (with pybind11)",
    author="Peter Spackman",
    license="GPLv3",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    cmake_install_dir="src/python/occ",
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.6",
)
