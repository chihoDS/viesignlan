from setuptools import find_packages, setup

NAME = 'vie-sign-lan'
VERSION = '0.1.0'
DESCRIPTION = ''
LICENSE = ''

with open('README.md', encoding="utf-8") as file: 
    description = file.read()

setup( 
    name=NAME, 
    version=VERSION, 
    description=DESCRIPTION, 
    author='Chi Ho', 
    author_email='chihods@gmail.com', 
    long_description=description, 
    long_description_content_type='text/markdown', 
    license=LICENSE,
    packages=find_packages('src'), 
    install_requires=[ 
        'numpy', 
        'pandas', 
    ], 
) 
