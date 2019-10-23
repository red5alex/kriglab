import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kriglab",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="Package for kriging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/red5alex/kriglab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2"
    ],
    python_requires='>=2.7',
)