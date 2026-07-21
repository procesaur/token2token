import setuptools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'requests',
    'transformers',
    'numpy',
    'tqdm',
]

setuptools.setup(
    name="token2token",
    version="1.0.2",
    author="Mihailo Škorić, based on {Kyubyong Park, Dongwoo Kim, Yo Joong Choe}",
    author_email="procesaur@gmail.com",
    description="Build and view token mappings between languages and tokenizers",
    install_requires=REQUIRED_PACKAGES,
    license='Apache License 2.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/procesaur/token2token",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12"
    ],
)
