# Intro to Deep Learning [December 10th, 2021]

Example scripts for running deep neural networks using Keras and TensorFlow2 in Python and R.

## Accessing files to run from the HPC (if using your own machine, skip this and see below!)

Example files are located in: `/home/bcbb_teaching_files/intro_deep_learning/`

NOTE: I suggest you either clone this GitHub repo or copy the HPC files to a folder in your local home directory! Please do not modify the python scripts in the course directory above. Normally, you would want to create your own conda environment with the specific packages you need for your own work - for those using the HPC for the class we will use a shared environment so we cut down on the number of redundant file copies.

#### Activate the conda environment so we can use Tensorflow and Keras on the HPC: 

##### First log into the HPC and check that you have `conda` available to you by typing:
`which conda`

##### *If nothing is returned* you need to initialize conda for your environment:
`/biocompace/condabin/conda init bash` (now log out and back in again and hopefully `conda` should be available to you)

#### Once you have `conda` available to you type:
`conda activate /home/bcbb_teaching_files/intro_deep_learning/envs`

#### If that does not work try the following:
`source /home/bcbb_teaching_files/intro_deep_learning/envs/bin/activate`

Hopefully, if things work, this should make Tensorflow v2.1.0 and other libraries available to you to run the examples. You should see the folder path appear at the start of your terminal. Test by running:

`python /home/bcbb_teaching_files/intro_deep_learning/1_keras_hello_world.py` and you should see a small model start to train and predict.

## Installing the environment on your own machine

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
