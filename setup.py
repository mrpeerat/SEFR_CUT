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
    'python-crfsuite',
    'pyahocorasick'
]
with open('README.md', 'r', encoding='utf-8-sig') as f:
    readme = f.read()

setup(
    name = 'SEFR_CUT',
    packages = ['sefr_cut'],
    include_package_data = True,
    version = '0.1dev0',
    install_requires = requirements,
    long_description = readme,
    long_description_content_type='text/markdown',
    license = 'MIT',
    description = '',
    author = '',
    author_email = '',
    url = 'https://github.com/mrpeerat/SEFR_CUT',
    download_url = '',
    keywords = ['thai word segmentation deep learning neural network development'],
    classifiers = [
        'Development Status :: 3 - Alpha'
    ],
)
