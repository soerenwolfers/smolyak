#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='smolyak',
      version='2',
      description='Accelerate scientific computations using Smolyak\'s algorithm',
      long_description=open('README.rst').read(),
      author='Soeren Wolfers',
      author_email='soeren.wolfers@gmail.com',
      packages=find_packages(exclude=['*tests','*docs']),#,'examples*']),
      install_requires=['numpy','scipy','swutil']
)
