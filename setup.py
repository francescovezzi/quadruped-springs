from setuptools import find_packages, setup

setup(
    name="quadruped_spring",
    packages=[package for package in find_packages() if package.startswith("quadruped_spring")],
    package_data={},
    install_requires=[
        "importlib-metadata<5.0",
        "pybullet==3.2.5",
        "torch==1.12",
        "scipy==1.7.3",
        "stable_baselines3==1.5.1.a8",
        "sb3-contrib==1.5.1.a8",
        "absl-py==1.0.0",
        "opencv-python==4.6.0.66",
    ],
    description="Gym environment pybullet based for simulating a quadruped",
    author="Francesco Vezzi",
    url="https://github.com/francescovezzi/quadruped_spring.git",
    author_email="f.vezzi.96@gmail.com",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning quadruped "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description="",
    long_description_content_type="text/markdown",
    version="0.1",
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
