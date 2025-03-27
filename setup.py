from setuptools import setup, find_packages

# Standard Library Imports
import os
import re
import time
# import logging
# import asyncio
# import threading
# from datetime import datetime
# import nest_asyncio


setup(
    name="conceptSetReview",  
    version="0.0.1",          
    author="EPAM",
    author_email="",
    description="Concept set review",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/OdyOSG/conceptSetReview",  
    packages=find_packages(),  
    install_requires=[
      "pyspark",
      #"numpy",
      "pandas",
      "pydantic",
      "langchain_openai"
    ]
)
