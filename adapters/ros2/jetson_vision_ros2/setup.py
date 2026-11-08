from glob import glob

from setuptools import find_packages, setup

package_name = "jetson_vision_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": ["vision_node = jetson_vision_ros2.node:main"]},
)
