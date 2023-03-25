import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="py0",
    version="1.0",
    url="https://github.com/ur-whitelab/py0",
    license='GPL v2',

    author="Mehrad Ansari, Rainier Barret, Andrew D White",
    author_email="mehrad.ans@gmail.com",

    description="Maximum Entropy Biasing of Compartmental Epidomiology Models",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=['numpy>=1.19.1', 'matplotlib==3.3.3', 'scipy', 'seaborn', 'tqdm', 'jupyter',
                      'maxent-infer', 'pandas', 'networkx==2.5', 'tensorflow==2.11.1', 'tensorflow_probability==0.11.1'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GPL v2 License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
