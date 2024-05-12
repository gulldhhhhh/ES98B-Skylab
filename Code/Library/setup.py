from setuptools import find_packages, setup

setup(
    name='pmsc_skylab', 
    packages=find_packages(include = ['__init__.py']),
    install_requires = ['numpy'],
    version='1.0.0',
    description='My first Python library',
    author=['Simon Coolsaet', 'Kyle Berwick', 'Gullveig Liang', 'Tanya Sanjeev', 'Leixin Xu'],
)