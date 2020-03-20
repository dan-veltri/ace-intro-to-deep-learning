# Intro to Deep Learning

* Update (Mar. 20, 2020) * I've added a start R script in the 'AMP Challenge' folder. Please note that the Python code is not compatible with R's version of Keras. Please see the [RStudio Keras Website](https://keras.rstudio.com) for details on using Keras with R.  
* Update (Mar. 19, 2020) * I've added a starter Python script for today's 'AMP Challenge' along with training and validation sets. The final test set will be added later!

Example scripts for running deep neural networks using Keras and TensorFlow2 in Python and R.

## Accessing files to run from the HPC

Example files are located in: `/home/bcbb_teaching_files/intro_deep_learning/`

NOTE: I suggest you either clone this GitHub repo or copy the HPC files to a folder in your local home directory!

Activate the conda environment: 

`conda activate /home/bcbb_teaching_files/intro_deep_learning/envs`

If that does not work try the following:

`source activate /home/bcbb_teaching_files/intro_deep_learning/envs`


This should make Tensorflow2 and other libraries available to you. At the moment, it appears an old version of conda is still installed causing the `conda activate` command to not work properly.

## Installing on your own machine

To run these you'll need python and the following packages installed. :
  * numpy 
  * scikit-learn
  * h5py
  * Pillow
  * matplotlib
  * tensorflow (v2 now includes keras)
  
I recommend installing packages using a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/). On a Linux machine, `pip` should work for the above packages but if you have Anaconda installed, you can easily use the `deep_learning_environment.yml` file to make a `deep_learning` environment via the command:
`conda create -f deep_learning_environment.yml`.

You can install to a specific directory using: `conda create --prefix ./envs -f deep_learning_environment.yml`  where `./envs` is the directory you want to install to. 

*Note For Mac Users! - I recommend installing `tensorflow` via Anaconda rather than `pip` (also applies to R users). You might also need to also install the `nomkl` package to prevent a multithreading bug in `numpy`.*
