try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'random_data',
    'version': '0.31',
    'packages': ['random_data', 'random_data.signals'],
    'install_requires': ['numpy', 'matplotlib', 'nose'],
    'author': 'Evan M. Davis',
    'author_email': 'emd@mit.edu',
    'url': '',
    'description': 'Python tools for random data analysis.'
}

setup(**config)
