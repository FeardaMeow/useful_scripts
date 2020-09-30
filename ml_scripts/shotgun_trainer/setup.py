import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "shotgun_trainer",
    version = "0.1.0",
    author = "Steven Hwang",
    author_email = "shwang1121@gmail.com",
    description = ("Trains many different models and plots their train and validation loss to understand model exploratory analysis."),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "",
    packages=find_packages(),
)