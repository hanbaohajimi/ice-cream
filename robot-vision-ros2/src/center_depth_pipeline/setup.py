from glob import glob
import os

from setuptools import find_packages, setup

package_name = "center_depth_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="YOLO center + depth pipeline for Angstrong RGBD",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detection_node = center_depth_pipeline.detection_node:main",
            "depth_node = center_depth_pipeline.depth_node:main",
            "grasp_node = center_depth_pipeline.grasp_node:main",
            "visualization_node = center_depth_pipeline.visualization_node:main",
            "pipeline_doctor = center_depth_pipeline.pipeline_doctor:main",
        ],
    },
)
