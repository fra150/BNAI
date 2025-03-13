from setuptools import setup, find_packages

setup(
    name="bnai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "pyyaml>=5.4.1",
        "tqdm>=4.62.3",
        "matplotlib>=3.4.3",
        "scikit-learn>=0.24.2",
    ],
)