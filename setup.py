from setuptools import setup

setup(
    name='MCTSLearn',
    version='0.0.1',
    packages=['mctslearn'],
    url='https://github.com/AntonOsika/MCTSLearn',
    license='LICENSE',
    author='Anton Osika',
    author_email='anton.osika@gmail.com',
    description='MCTS using ML library',
    install_requires=[
        'gym>=0.10.5',
        'numpy>=1.15.0',
        'setuptools>=40.0.0',
        'msgpack_python>=0.5.6',
        'msgpack_numpy>=0.4.4.1',
        'torch>=0.4.0',
        'tqdm>=4.28.1',
        'ghp-import',
    ],
)
