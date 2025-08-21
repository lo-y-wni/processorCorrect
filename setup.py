#!/usr/bin/env python

from setuptools import setup, find_packages
import os

setup(
    name='processorCorrect',
    version='1.0',
    description='A package for Dual PRF/Staggered PRF velocity correction.',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/lo-y-wni/processorCorrect',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'datetime'
    ],
    python_requires='>=3.7',
    include_package_data=True,
    license='MIT',  # Update as appropriate
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)
