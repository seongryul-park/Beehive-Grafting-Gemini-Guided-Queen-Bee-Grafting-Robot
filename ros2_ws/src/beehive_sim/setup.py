import os
from glob import glob

from setuptools import find_packages, setup

package_name = "beehive_sim"


def _data_files():
    files = [
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    # launch / worlds
    files.append(("share/" + package_name + "/launch", glob("launch/*.launch.py")))
    files.append(("share/" + package_name + "/worlds", glob("worlds/*.sdf")))
    # gazebo models (preserve folder structure, including texture binaries)
    for path in glob("models/**/*", recursive=True):
        if os.path.isfile(path):
            dest = os.path.join("share", package_name, os.path.dirname(path))
            files.append((dest, [path]))
    return files


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=_data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Seongryul",
    maintainer_email="seongryul.digipen@gmail.com",
    description="Gazebo bringup + live camera -> existing beehive vision pipeline.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_node = beehive_sim.vision_node:main",
        ],
    },
)
