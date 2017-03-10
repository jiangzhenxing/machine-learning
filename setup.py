from setuptools import find_packages
from distutils.core import setup
import setuptools

setup(  name='machine-learning',
        version = '0.1',
        packages = find_packages('src'),
        package_dir = {'':'src'},
#        data_files = [('config', ['etc/config.txt']),]
     )
