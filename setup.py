import setuptools

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name='BERTSimilarWords',
    version='0.0.8',
    description="Find Similar Words using BERT",
    keywords="BERT NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    author="Pahalavan R D",
    author_email='rdpahalavan24@gmail.com',
    packages=['BERTSimilarWords'],
    python_requires='>=3.7.0',
    url='https://github.com/rdpahalavan/BERTSimilarWords',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)