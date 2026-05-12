from setuptools import find_packages, setup

package_name = "object_base_logger"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/head_ingestion.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "websockets"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="ROS2 node for realtime detection console logging",
    license="Apache-2.0",
    tests_require=["pytest"],
    # 使用 scripts= 而非 console_scripts，避免 symlink-install 下
    # importlib.metadata.PackageNotFoundError: object-base-logger
    scripts=["scripts/base_logger_node"],
)
