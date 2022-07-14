This is a forked implementation of https://arxiv.org/pdf/1909.01807.pdf . All code referenced and/or forked is licensed with [Creative Commons Attribution 4.0 International Public License]("https://creativecommons.org/licenses/by/4.0/).    

# How to Build
First install Microsoft Visual C++ 2014 or newer. Neuralcoref needs this to build on Windows at least. 

Next, install conda. I used miniconda since we just need the conda cli. We need conda for managing specific python versions. We have to use an old version of spacy since Neuralcoref  does not support spacy 3. We have to install the spacy model from conda-forge, an alternative source, since the model version we need for python 3.6 with spacy 2 is not in the default conda repos.

In this folder run
```
conda env create -f environment.yml
conda env activate tripletParser
```
This will install all the required packages and place you in the tripletParser environment with the right Python version.


### Alternative Build Option

Create a Python 3.6 installation. Conda is useful for this, but you can use a default system installation of python 3.6 as well.
```
conda create -n KGPARSER python=3.6
conda activate KGPARSER
```
Make sure you are installing while within the activated environment.  
```
pip install -r .\requirements.txt
conda install -c conda-forge spacy-model-en_core_web_sm=2.1.0
pip install pdfminer.six
```

This essentially does the same thing as above but installs through pip inside the conda env instead of conda.