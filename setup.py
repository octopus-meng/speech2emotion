from setuptools import find_packages
from distutils.core import setup

setup(
    name='ser',
    version='0.1.0',
    author='',
    license="MIT",
    packages=find_packages(),
    author_email='',
    description='Speech and Text Emotion Recognition using Large Language Models',
    install_requires=[
        'openai',
        'httpx',
    ]
)
