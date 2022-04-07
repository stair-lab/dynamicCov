import os
import io
import sys
from setuptools import find_packages, setup, Command

with open('./requirements.txt') as rf:
      requirements = rf.read().splitlines()

with open('README.md') as rf:
    readme = rf.read()

with open('LICENSE') as rf:
    license = rf.read()


setup(name='snscov',
      version='0.2',
      description='Non-convex framework for structured non-stationary covariance recovery',
      author='Katherine Tsai, Mladen Kolar, Oluwasanmi Koyejo',
      author_email='kt14@illinois.edu',
      long_description=readme,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python3.7',
          'License :: OSI Approved :: MIT License',
          'Operating System :: Unix',
          'Operating System :: iOS'
          ],
    packages=['snscov'],
    license=license,
    install_requires=requirements)
