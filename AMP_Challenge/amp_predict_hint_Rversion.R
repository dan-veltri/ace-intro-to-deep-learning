#!/usr/bin/env Rscript
# amp_predict_template.R - Use this to get started building your predictive model
#
# amps_file and decoys_file should point to respective AMP and DECOY text files
# where sequences are each on their own line.
#
# Be sure to save your final model with a ".h5" extension for testing! 
#============================================================================================

library(keras)
library(pROC)

# IF you're using RStudio and your files are in the same folder as the script -
# Select "Session" -> "Set Working Directory" -> "To Source File Location"
# Otherwise these files might not be found and you'll need to add the FULL path to their location! 
amps_train_file <- 'AMP.train.txt'
decoys_train_file <- 'DECOY.train.txt'

amps_test_file <- 'AMP.eval.txt'
decoys_test_file <- 'DECOY.eval.txt'

#Load sequences (20 amino acids converted to numbers) into x_train, and binary responses/labels into y_train (1=AMP,0=Decoy)
max_length <- 200 # No sequences are longer than 200 amino acids!
aminos <- c('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y')
aa2int <- function(seq){match(strsplit(seq,'')[[1]],aminos)}

getSeqs <- function(fileName){
  fileIn <- file(fileName,"r")
  i <- 1
  xlist <- list()
  while(TRUE){
    seq_line <- readLines(fileIn,n=1)
    if(length(seq_line) == 0){ break }
    xlist[[i]] <- aa2int(seq_line)
    i <- i + 1
  }
  close(fileIn)
  return(xlist)
}

# Load and pad training data
amps_train <- getSeqs(amps_train_file)
decoys_train <- getSeqs(decoys_train_file)
x_train <- pad_sequences(c(amps_train,decoys_train),maxlen=max_length) # make equal length with 0's
y_train <- c(rep(1,length(amps_train)),rep(0,length(decoys_train)))

# Shuffle training data
idx <- sample(length(y_train))
x_train <- x_train[idx,]
y_train <- y_train[idx]

# Load and pad test data
amps_test <- getSeqs(amps_test_file)
decoys_test <- getSeqs(decoys_test_file)
x_test <- pad_sequences(c(amps_test,decoys_test),maxlen=max_length) # make equal length with 0's
y_test <- c(rep(1,length(amps_test)),rep(0,length(decoys_test)))


print("Compiling model...")
model <-  keras_model_sequential()
model %>%
  layer_embedding(21, ???, input_length=max_length) %>%
  
# --- YOUR MODEL HERE --- %>%
  
  layer_dense(1, activation=???)

model %>% compile(loss=???, optimizer='adam', metrics=c('accuracy'))
model %>% fit(x_train, y_train, epochs=???, batch_size=???, verbose=1)

print("Prediction training performance...")
preds <- model %>% predict_classes(x_train)

#Calculate performance stats
cm <- data.frame(tn=0,fn=0,fp=0,tp=0) #confusion matrix
cm[, c("tn","fn","fp","tp")] <- table(Actual=y_train,Predicted=preds)
sens <- cm$tp / (cm$tp + cm$fn) * 100.0
spec <- cm$tn / (cm$tn + cm$fp) * 100.0
prec <- cm$tp / (cm$tp + cm$fp) * 100.0
acc <- (cm$tp + cm$tn) / (cm$tn + cm$fp + cm$fn + cm$tp) * 100.0
auroc <- auc(y_train, preds[,1]) #from pROC package

#For MCC here we use as.numerics to prevent overflows, R-ints are still 32bit!
mcc_numerator <- as.numeric(cm$tp * cm$tn) - as.numeric(cm$fp * cm$fn) 
mcc_denominator <- sqrt(as.numeric(cm$tp + cm$fp) * as.numeric(cm$tp + cm$fn) * as.numeric(cm$tn + cm$fp) * as.numeric(cm$tn + cm$fn)) 

print(paste0("TP: ", cm$tp, ", TN: ", cm$tn, ", FP: ", cm$fp, ", FN: ", cm$fn))
print(paste0("Sensitivity: ", round(sens,4)))
print(paste0("Specificity: ", round(spec,4)))
print(paste0("Accuracy: ", round(acc,4)))
print(paste0("MCC: ", round(mcc_numerator/mcc_denominator,4)))
print(paste0("Area Under ROC: ", round(auroc,4)))
print(paste0("Precision: ", round(prec,4)))

# Save your model
save_model_hdf5(model, "my_team_model.h5", overwrite = TRUE)

# END PROGRAM


