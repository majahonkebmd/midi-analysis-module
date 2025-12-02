# setup.py
from setuptools import setup, find_packages

setup(
    name="midi-analysis-module",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pretty-midi>=0.2.10",
        "mido>=1.2.0",
        "python-rtmidi>=1.4.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "fastdtw>=0.3.4",
    ],
    author="Your Name",
    description="Educational MIDI analysis tool for piano pedagogy",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)