"""Copyright (C) SquareFactory SA - All Rights Reserved.
This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
from setuptools import find_packages, setup

with open("retina/train_requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="retina",
    packages=find_packages(include=["retina", "retina.*"]),
    package_data={"retina": ["config.yml"]},
    version="0.0.1",
    description="Face detector and pixelizer, isquare usage example",
    author="SquareFactory SA",
    license="MIT",
    requires_python=">=3.8",
    install_requires=requirements,
    setup_requires=[],
)
