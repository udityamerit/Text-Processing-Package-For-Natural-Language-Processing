import setuptools

with open('Readme.md') as fp:
    long_description = fp.read()

# Use splitlines() to create a list of requirements
with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setuptools.setup(
    name='nlp_text_preprocessing',
    include_package_data=True,
    version='0.0.2',
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
    
    python_requires='>=3.10',
)
