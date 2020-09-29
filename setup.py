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
    packages = ['sefr_cut','sefr_cut.deepcut','sefr_cut.model','sefr_cut.variable','sefr_cut.weight','sefr_cut.deepcut.weight'],
    include_package_data = True,
    package_data={"sefr_cut": ['model/*','variable/*','weight/*','deepcut/weight/*']},
    version = '0.2',
    install_requires = requirements,
    long_description = readme,
    long_description_content_type='text/markdown',
    license = 'MIT',
    description = 'Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble (EMNLP2020)',
    author = 'mrpeerat',
    author_email = 'peerat.l_s19@vistec.ac.th',
    url = 'https://github.com/mrpeerat/SEFR_CUT',
    download_url = '',
    keywords = ['thai word segmentation deep learning machine learning neural network development'],
    classifiers = [
        'Development Status :: 3 - Alpha'
    ],
)
