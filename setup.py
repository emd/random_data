try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'random_data',
    'version': '0.4',
    'packages': ['random_data', 'random_data.signals', 'random_data.spectra'],
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'nose'],
    'author': 'Evan M. Davis',
    'author_email': 'emd@mit.edu',
    'url': '',
    'description': 'Python tools for random data analysis.'
}

setup(**config)
