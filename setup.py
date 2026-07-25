import setuptools

with open('Readme.md', encoding='utf-8') as fp:
    long_description = fp.read()

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setuptools.setup(
    name='nlp_text_preprocessing',
    include_package_data=True,
    version='0.1.0',
    author='Uditya Narayan Tiwari',
    author_email='tiwarimerit@gmail.com',
    description='This is a Text Processing Package For NLP',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=requirements,
    # explicit instruction to include data files in the package
    package_data={'nlp_text_preprocessing': ['data/*.json']}, 
    python_requires='>=3.9',
)