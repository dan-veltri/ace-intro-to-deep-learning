# Intro to Deep Learning

Example scripts for running deep neural networks using Keras and TensorFlow2 in Python.

## Accessing files to run from the HPC

Example files are located in: `/home/bcbb_teaching_files/intro_deep_learning/`

NOTE: I suggest you either clone this GitHub repo or copy the HPC files to a folder in your local home directory!

Activate the conda environment: `source activate /home/bcbb_teaching_files/intro_deep_learning/envs`
This should make Tensorflow2 and other libraries available to you.

## Installing on your own machine

To run these you'll need python and the following pip packages installed. As Keras updates a lot (often breaking older code syntax) note the package version numbers the code was tested on. I recommend installing packages using a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
  * numpy 
  * scikit-learn
  * h5py
  * Pillow
  * matplotlib
  * tensorflow (ver. 1)
  * keras
  
*Note For Mac Users: I recommend installing `tensorflow` via Anaconda rather than PIP (also applies to R users). You might also need to also install the `nomkl` package to prevent a multithreading bug in `numpy`.*
