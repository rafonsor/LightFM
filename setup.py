from setuptools import setup

try:
    from lightfm import __version__
except Exception:
    __version__ = open('lightfm/__version__.py').read().split()[-1].strip("'\"")


requirements = open('requirements.txt', 'rt').read().split('\n')

setup(
    name='lightfm-pytorch',
    packages=['lightfm'],
    version=__version__,
    license='MIT',
    description='A PyTorch re-implementation of LightFM Hybrid Recommendation model.',
    author='Rafael Afonso Rodrigues',
    author_email='rafael.ar@live.be',
    keywords=['LightFM', 'Recommender', 'collaborative-filtering', 'ML'],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
)
