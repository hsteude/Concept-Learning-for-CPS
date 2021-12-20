from setuptools import setup, find_packages

setup(
    name='concept-learning-cps',
    version='0.0.1',
    description='Code for paper',
    author='Alexander Windmann and Henrik Steude',
    author_email='henrik.steude@gmail.com',
    url='https://github.com/hsteude/...',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'pandas',
        'numpy',
        'scipy',
        'loguru',
        'torch',
        'pytorch_lightning',
        'joblib',
        'tqdm',
        'scikit-learn',
        'plotly',
        'fastparquet',
        'matplotlib',
        'kaleido'
        ],
)
