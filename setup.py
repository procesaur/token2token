import setuptools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'requests',
    'wget',
    'numpy',
    'tqdm',
]

setuptools.setup(
    name="word2word",
    version="1.0.0",
    author="Mihailo Škorić, based on {Kyubyong Park, Dongwoo Kim, Yo Joong Choe}",
    author_email="procesaur@gmail.com",
    description="Build and view token mappings between languages and tokenizers",
    install_requires=REQUIRED_PACKAGES,
    license='Apache License 2.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kakaobrain/word2word",
    packages=setuptools.find_packages(),
    package_data={'token2token': ['token2token/supporting_languages.txt']},
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
