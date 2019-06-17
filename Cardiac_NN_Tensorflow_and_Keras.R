## In my simulated cardio_nn_dataset there are 27 columns,
# each of which entries take on 0 or 1 if absent or present; 
# 0 = alive, 1 = dead. Besides [,1] which is a placeholder.

## This dataset was created using summary statistics of each column of the real Cardiology dataset
# I worked with daily at the University of Utah School of Medicine.
# This dataset is in fact statistically indistinguishable from the real dataset.
## I can disuss the ease of coding up such realistic simulated datasets,
# and their uses for data validation and determination of the robustness of statistical summary calculations,
# further if requested.

## Everything is run in R, with the tensorflow and keras machine learning libraries
# accessed via special APIs through R. ##

## Load Worthington's "ipak" function:
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

## Give "packages" the names of the packages you'll use in your data tasks:
packages <- c("janitor","dplyr","tensorflow","keras")

## Run "ipak" and open all the packages:
ipak(packages)

## Our goal is to classify which patients are dead, and which are alive.
# Most Cardiology patients are alive, but about 10% of patients are dead.
# We would, ideally, like to figure out what all the dead ones have in common 
# before they die in the future, using 26 common diseases associated with heart rhythm problems.

## Read in the simulated data:
cardio_nn_dataset <- read.csv("fake_cardio_death_data_1.csv")

## Use R package "janitor" to "clean" column names.
# (Some ML packages get glitchy over R column names).
cardio_nn_dataset <- cardio_nn_dataset %>% clean_names()

## Remove a useless column:
cardio_nn_dataset <- cardio_nn_dataset %>% select(-x)

## Confirm the data transformation:
dim(cardio_nn_dataset)
names(cardio_nn_dataset)
str(cardio_nn_dataset)

## A cheap trick to get variable "sim_gender" in line with other binary variables.
# For neural networks, all binary predictor varibales should be 0s and 1s.
cardio_nn_dataset$sim_gender <- as.integer(cardio_nn_dataset$sim_gender) - 1
cardio_nn_dataset$sim_gender <- as.integer(cardio_nn_dataset$sim_gender)
str(cardio_nn_dataset$sim_gender)

## For neural networks, turn your dataframe into a matrix
# and get rid of the dimnames.
cardio_nn_dataset <- as.matrix(cardio_nn_dataset)
dimnames(cardio_nn_dataset) <- NULL

## Determine sample size index.
ind <- sample(2, nrow(cardio_nn_dataset), replace=TRUE, prob=c(0.80, 0.20))

## Split the `cardio_nn_dataset data into training and test datasets.
death.training <- cardio_nn_dataset[ind==1, 1:25]
death.test <- cardio_nn_dataset[ind==2, 1:25]

# Split the class attribute to specify what the target variable we're classifying
death.trainingtarget <- cardio_nn_dataset[ind==1, 26]
death.testtarget <- cardio_nn_dataset[ind==2, 26]

## Now do a one-hot encoding and
# convert the target variables back into categories.
death.trainLabels <- to_categorical(death.trainingtarget)
death.testLables <-  to_categorical(death.testtarget)

## Initialize a model:
model <- keras_model_sequential() 

## Try 2/3 size of the input layer + size of output layer for
# the number of hidden neurons.
## Add input, hidden layers, and an output layer to the model:
# Input shape is 26 because there are 26 variables.
model %>% layer_dense(units = 32, activation = 'relu', input_shape = c(25)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense (units = 8, activation = "relu") %>% 
  layer_dropout (rate = 0.2) %>%
  layer_dense(units = 2, activation = 'softmax')

## Compile the model:
history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

## Fit the model; 
fit <- model %>% fit(
  death.training, 
  death.trainLabels, 
  epochs = 100, 
  batch_size = 5, 
  validation_split = 0.2,
  verbose = 1
)

## Plot the final modeling results:
plot(fit)

## Take a look at the final model tensorflow fitted:
summary(model)

## Evaluate:
model %>% evaluate(death.test,death.testLables, batch_size = 32)

## Predict the outputs using the test sample:
model %>% predict(death.test, batch_size = 32)
model %>% predict(death.test, steps = 30)

## Optional stuff ##
# Get model configuration:
#get_config(model)

## Get layer configuration:
#get_layer(model, index = 1)

## List the model's layers:
#model$layers

## List the input tensors:
#model$inputs

## List the output tensors:
#model$outputs

## Predict the classes for the test data
classes <- model %>% predict_classes(death.test, batch_size = 128)

## Make a confusion matrix:
table(death.testtarget,classes)

## Use the confusion matrix to calculate model performance metrics:


## Accuracy
(3285+0)/(3285+0+432+1)
# [1] 0.8835395

## Precision
3285/(3285+1)
# [1] 0.9996957

## Specificity
# We see that we need to do some hyperparameter
# tuning to figure out why the specificity is 0.
0/(0+1).
# [1] 0

## OVERALL FINDINGS ##
## ~88% Accuracy and ~99% precision but 0% specificity.
## There is always a trade-off between accuracy, precision and specificity,
# but the best neural networks is the one that gives relatively high values of all 3 metrics.
# Using under-sampling of the most common target variable class,
# or over-sampling of the least common target variable class, for example the ADASyn algorithm,
# should improve the overall performance of the neural network.

## For a first neural network in tensorflow, this is fast code that mostly does what we want,
# but will need hyperparameter tuning and balanced target variable classes to make any clinically relevant decisions.
