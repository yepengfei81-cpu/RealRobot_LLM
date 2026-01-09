from __future__ import annotations

from setuptools import find_packages, setup

package_name = 'airbot_grpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='discover',
    maintainer_email='discover@163.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
)
