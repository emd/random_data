try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'random_data',
    'version': '0.1',
    'packages': ['random_data'],
    'install_requires': ['nose'],
    'author': 'Evan M. Davis',
    'author_email': 'emd@mit.edu',
    'url': '',
    'description': 'Python tools for random data analysis.'
}

setup(**config)
