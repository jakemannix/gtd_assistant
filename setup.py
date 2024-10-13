import os
from setuptools import setup, find_packages

# Read requirements from files
def read_requirements(filename):
    requirements = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('-r '):
                    # If the line is a file reference, read that file
                    requirements += read_requirements(line.split(' ')[1])
                else:
                    requirements.append(line)
    return requirements

setup(
    name="gtd_assistant",
    version="0.1",
    packages=find_packages(exclude=['tests']),
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        "dev": read_requirements('requirements-dev.txt'),
    },
)
