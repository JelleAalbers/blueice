#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()

setup(name='blueice',
      version='0.2.0',
      description='Build Likelihoods Using Efficient Interpolations from monte-Carlo generated Events',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/JelleAalbers/blueice',
      packages=['blueice'],
      package_dir={'blueice': 'blueice'},
      install_requires=requirements,
      license="MIT",
      zip_safe=False,
      keywords='blueice',
      classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
)