#!coding:utf8
from distutils.core import *
import gttrack
current_version = gttrack.VERSION 

if __name__ == '__main__':
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setup(
        name='gttrack',
        version=current_version,
        packages=['gttrack'],
        url='http://www.geetest.com',
        license='',
        author='Geetest',
        author_email='admin@geetest.com',
        description='Rules updated for recent attack',
        install_requires=required,

        # package_data={'gttrack': ['pkl/*.pkl']},
        # data_files=[
        # ('gttrack/pkl',['gttrack/pkl/RandomForest.pkl'])
        # ]
    )
