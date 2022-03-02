from setuptools import find_packages, setup

setup(
    name="quadruped_spring",
    packages=[package for package in find_packages() if package.startswith("quadruped_spring")],
    package_data={},
    install_requires=[],
    description="",
    author="",
    url="",
    author_email="",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
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
