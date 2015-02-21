try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'projectname',
    'version': '0.1',
    'packages': ['NAME'],
    'install_requires': ['nose'],
    'author': 'Evan M. Davis',
    'author_email': 'emd@mit.edu',
    'url': 'URL to get it at.',
    'description': 'My Project'
}

setup(**config)
