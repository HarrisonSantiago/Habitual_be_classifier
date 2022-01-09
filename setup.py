import setuptools

setuptools.setup(
    name="habitual_be_classifier",
    version="0.1.0",
    author="Harrison Santiago",
    author_email="harrisonsantiago4@gmail.com",
    description="Disambiguating habitual from non-habitual be",
    url="https://github.com/HarrisonSantiago/Habitual_be_classifier",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'nltk',
        'sklearn',
        'nlpaug'
    ],
)