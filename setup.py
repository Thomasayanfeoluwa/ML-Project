from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a requirements file and returns a list of packages.
    '''
    requirements= []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements
    


setup(
    name='MLProject',
    version='0.1.0',
    author='Thomas',
    author_email='ayanfeoluwadegoke@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)