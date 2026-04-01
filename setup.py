"""Setup script for QuBiS-HiQ."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qubis-hiq",
    version="1.0.0",
    author="Ahmed Anees",
    description="Quantum Biomimetic Sequence encoding for High-throughput in-silico Quantum computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedanees-m/QuBiS-HiQ",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "pylint>=2.15.0",
        ],
    },
)
