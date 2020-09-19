#! /usr/bin/env python
"""
Thai word Segmentation using Convolutional Neural Network
"""

from setuptools import setup

requirements = [
    'tensorflow>=2.0.0',
    'pandas',
    'scipy',
    'numpy',
    'scikit-learn',
    'pycrfsuite',
    'pyahocorasick'
]

setup(
    name = 'SEFR_CUT',
    packages = ['sefr_cut'],
    include_package_data = True,
    version = '0.1dev0',
    install_requires = requirements,
    license = 'MIT',
    description = '',
    author = '',
    author_email = '',
    url = '',
    download_url = '',
    keywords = ['thai word segmentation deep learning neural network development'],
    classifiers = [
        'Development Status :: 3 - Alpha'
    ],
)
