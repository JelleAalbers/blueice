#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()
# test_requirements = requirements  # + ['flake8', 'tox', 'coverage','bumpversion']

setup(name='wimpy',
      version='0.1',
      description='Simple XENON1T S1/S2 response Monte Carlo',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/XENON1T/wimpy',
      packages=['wimpy', 'wimpy.models'],
      package_dir={'wimpy': 'wimpy'},
      install_requires=requirements,
      license="MIT",
      zip_safe=False,
      keywords='wimpy',
      classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
      # test_suite='tests',
      # tests_require=test_requirements
)
