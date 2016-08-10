#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()

setup(name='blipflip',
      version='0.2.0',
      description='Build Likelihoods from Interpolated Pdfs For LImit settting and other Parametric inference',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/JelleAalbers/blipflip',
      packages=['blipflip'],
      package_dir={'blipflip': 'blipflip'},
      install_requires=requirements,
      license="MIT",
      zip_safe=False,
      keywords='blipflip',
      classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
)