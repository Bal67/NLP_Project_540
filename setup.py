from setuptools import setup, find_packages

setup(
    name='sentiment_analysis_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'tensorflow',
        'seaborn',
        'matplotlib',
        'nltk',
        'emoji',
        'joblib'
    ],
)

