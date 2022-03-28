# setup.py

# import setuptools
# setuptools.setup()

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="automatize",
    version="0.1.0",
    author="tarlis",
    author_email="tarlis@tarlis.com.br",
    description="Automatize: Multi-Aspect Trajectory Data Mining Tool Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ttportela/automatize",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ),
)