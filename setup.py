from setuptools import setup
import os
from glob import glob

package_name = 'visual_multi_crop_row_navigation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # This creates the lib/<pkg>/ directory that ros2 launch requires:
        (os.path.join('lib', package_name), glob('visual_multi_crop_row_navigation/vs_navigator.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alireza',
    maintainer_email='alireza.ahmadi@uni-bonn.todo',
    description='phenobot multicrop row visual servoing navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vs_navigator = visual_multi_crop_row_navigation.vs_navigator:main',
        ],
    },
)