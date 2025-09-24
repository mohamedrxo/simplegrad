from setuptools import setup, find_packages

setup(
    name="simplegrad",               # package name on PyPI
    version="0.1.0",
    description="A minimal autograd engine built from scratch",
    author="Mohamed Rachoum",
    packages=find_packages(),        # finds all folders with __init__.py
    install_requires=["numpy"],      # dependencies
    python_requires=">=3.7",         # minimum Python version
)
