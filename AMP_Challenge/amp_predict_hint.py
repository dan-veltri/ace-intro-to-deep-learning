#!/usr/bin/env python
"""
 amp_predict_template.py - Use this to get started building your predictive model

 amps_file and decoys_file should point to respective AMP and DECOY FASTA files.

 Be sure to save your final model with a ".h5" extension for testing later! 
"""

from __future__ import print_function #python3 printing

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.utils import shuffle

import tensorflow as tf # we can now access tf.keras as needed
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model # needed to load saved models
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.layers import  .... YOUR IMPORTS HERE ....


#Paths to training files
amps_train_file = 'AMP.train.txt'
amps_test_file = 'AMP.eval.txt'
decoys_train_file = 'DECOY.train.txt'
decoys_test_file = 'DECOY.eval.txt'

X_train = []
y_train = []
X_test = []
y_test = []

max_length = 200 # No sequences are longer than 200 amino acids!
amino_acids = "XACDEFGHIKLMNPQRSTVWY" # Give X index of 0, a padding character
aa2int = dict((c, i) for i, c in enumerate(amino_acids))

print("Encoding training/testing sequences...")

# A function to load in our peptide sequences
def load_seqs(X, y, seq_file, label=0):
	with open(seq_file) as fp:
		seq = fp.readline().rstrip()
		while seq:
			X.append([aa2int[aa] for aa in seq.rstrip().upper()])
			y.append(label)
			seq = fp.readline()
    	
	return X, y
	
# We'll give AMPs the label 1, and Decoys the label 0
X_train, y_train = load_seqs(X_train, y_train, amps_train_file, 1)
X_train, y_train = load_seqs(X_train, y_train, decoys_train_file, 0)
X_test, y_test = load_seqs(X_test, y_test, amps_train_file, 1)
X_test, y_test = load_seqs(X_test, y_test, decoys_train_file, 0)

# Pad input sequences with 0's in front and shuffle
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_train, maxlen=max_length)
X_train, y_train = shuffle(X_train, np.array(y_train))
y_test = np.array(y_test)




model =  .... YOUR CODE HERE ....





print("\nPredicting validation performance...")
pred_values = model.predict(X_test)
pred_classes = np.rint(pred_values) #round up or down at 0.5
true_classes = np.array(y_test)

#Get performance stats    
tn, fp, fn, tp = confusion_matrix(y_test,pred_classes).ravel()
roc = roc_auc_score(true_classes,pred_values) * 100.0
mcc = matthews_corrcoef(true_classes,pred_classes)
acc = (tp + tn) / (tn + fp + fn + tp + 0.0) * 100.0
sens = tp / (tp + fn + 0.0) * 100.0
spec = tn / (tn + fp + 0.0) * 100.0
prec = tp / (tp + fp + 0.0) * 100.0
print("\nModel Performance:\n\nTP: {}\tTN: {}\nFP: {}\tFN: {}\n\nSensitivity: {}%\nSpecificity: {}%\nAccuracy: {}%\nMCC: {}\nArea Under ROC: {}%\nPrecision: {}%".format(tp,tn,fp,fn,np.round(sens,2),np.round(spec,2),np.round(acc,2),np.round(mcc,4),np.round(roc,2),np.round(prec,2)))

#Save your model
model.save('my_team_model.h5')

# END PROGRAM