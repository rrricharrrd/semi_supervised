from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='semisupervised',
      version='0.1',
      description='Some semi-supervised learning algorithms',
      url='https://github.com/rrricharrrd/semi_supervised',
      author='Richard Harris',
      #author_email='',
      license='MIT',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3'
      ],
      packages=['semisupervised', 'semisupervised.puAdapter', 'semisupervised.ssgmm'],
      install_requires=[
          'numpy>=1.6.1',
          'scipy>=0.9',
          'scikit-learn>=0.17.1',
      ],
      scripts=[],
      zip_safe=False)

