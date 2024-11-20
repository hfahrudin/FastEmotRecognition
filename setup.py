import setuptools

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="facex", # Replace with your own username
    version="1.1.0",
    author="Hasby Fahrudin",
    author_email="fahrudinhasby12@gmail.com",
    description="Lightweight, High-Performance Facial Expression Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hfahrudin/facex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    install_requires=[
        'scikit-image>=0.16.2',
        'tensorflow>=2.0.0',
        'opencv-contrib-python>=4.10.0.84',
    ],
    python_requires='>=3.6',
)
