from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="simplegrade",
    version="0.1.2",
    author="Mohamed Rachoum",
    description="A lightweight PyTorch-like autograd library built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohamedrxo/simplegrad",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.7",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)