## Use Support Vector Machine models to predict whether patients are dead or alive.
# I use SVMs here because they're suprisingly fast compared to neural networks and 
# it's often easier to make sense of the results compared to neural networks.

## Load Worthington's "ipak" function:
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

## Give "packages" the names of the packages you'll use in your data tasks:
packages <- c("tidyverse","plyr","dplyr","magrittr","skimr","janitor","msm")

## Run "ipak" and open all the packages:
ipak(packages)

# Read in the simulated data:
cardio_nn_dataset <- read.csv("fake_cardio_death_data_1.csv")

## Use R package "janitor" to "clean" column names.
# (Some ML packages get glitchy over R column names).
cardio_nn_dataset <- cardio_nn_dataset %>% clean_names()

## If these give different names, some columns need to be renamed for ease of handling by R package "caret".
names(cardio_nn_dataset)
make.names(names(cardio_nn_dataset))

dim(cardio_nn_dataset)
str(cardio_nn_dataset)

glimpse(cardio_nn_dataset)

## Make sure all the factors are factors:
shouldBeFactor <- c("sim_drugs","sim_alcoholism","sim_alcoholism","sim_anemia","sim_bloodloss","sim_obesity",
                    "sim_coagulopathy","sim_rheumatic","sim_tumor","sim_mets","sim_lymphoma","sim_hiv","sim_pud",
                    "sim_liver","sim_renal","sim_hypothyroid","sim_diab","sim_pulmonary","sim_neuroother",
                    "sim_paralysis","sim_htn","sim_pvd","sim_phtn","sim_valvular","sim_chf","sim_death_calc")

for(v in shouldBeFactor) {
  cardio_nn_dataset[[v]] <- as.factor(cardio_nn_dataset[[v]])
}

## Get rid of this "x" column:
cardio_nn_dataset <- cardio_nn_dataset %>% select(-x)

## Inspect the outputs of your data transformation:
names(cardio_nn_dataset)

## Make the new training and test dataset subsets from the new over-sampled dataset:
library(caret)

## Set seed to reproduce results obtained later on:
set.seed(1234)

## Use caret's native partition function to split the dataset:
training.sample.1 <- cardio_nn_dataset$sim_death_calc %>% 
  createDataPartition(p = 0.80, list = FALSE)
train.data.1  <- cardio_nn_dataset[training.sample.1, ]
test.data.1 <- cardio_nn_dataset[-training.sample.1, ]

## Give levels of the target variable names so "caret" doesn't get confused during nn computations.
levels(train.data.1$sim_death_calc) <- c("alive", "dead")
levels(test.data.1$sim_death_calc) <- c("alive", "dead")

## These are the binary classification types we have to choose from:
# C-svc, nu-svc and C-bsvc

# We'll run a bunch that could be applicable for the data we have.
require(kernlab)
## We have a choice to make between the type of method we use for classification: C-svc or nu-svc.
# C can be from 0 to infinity, but nu controls the number of support vectors
# and is regularized between 0 and 1.
# This means that nu often achieves slightly better results, but can take longer to run.
svp.1 <- ksvm(sim_death_calc ~ ., data = train.data.1, type = "nu-svc", C = 1,
              kernel = "vanilladot", prob.model=TRUE, cross=10) # Simplest kernel function.

svp.2 <- ksvm(sim_death_calc ~ ., data = train.data.1, type = "nu-svc", C = 1,
              kernel = "rbfdot", prob.model=TRUE, cross=10) # General purpose kernel.

svp.3 <- ksvm(sim_death_calc ~ ., data = train.data.1, type = "nu-svc", C = 1,
              kernel = "polydot", prob.model=TRUE, cross=10) # Good for image classification.

svp.4 <- ksvm(sim_death_calc ~ ., data = train.data.1, type = "nu-svc", C = 1,
              kernel = "tanhdot", prob.model=TRUE, cross=10) # A proxy kernel for neural networks.

svp.5 <- ksvm(sim_death_calc ~ ., data = train.data.1, type = "nu-svc", C = 1,
              kernel = "laplacedot", prob.model=TRUE, cross=10) # General purpose kernel.

## Not used:
# "besseldot" General purpose kernel, insanely slow on this data.
# "anovadot", "splinedot" and "stringdot" are great for regression tasks.

## Let's predict some data and see how accurate the final SVM classifier is:
# We compare the success of the training data vs the success of classifier on the test data.
# We choose the SVM with the best of each success rate.

## SVM model 1 is OK, but if we're predicting clinical outcomes?
# It's not great at all.
pred_train.1 <- predict(svp.1,train.data.1)
mean(pred_train.1==train.data.1$sim_death_calc)
pred_test.1 <- predict(svp.1,test.data.1)
mean(pred_test.1==test.data.1$sim_death_calc)

## SVM model 2, is the clear number 2 here, with the "rbfdot" kernel:
pred_train.2 <- predict(svp.2,train.data.1)
mean(pred_train.2==train.data.1$sim_death_calc)
pred_test.2 <- predict(svp.2,test.data.1)
mean(pred_test.2==test.data.1$sim_death_calc)

## SVM 3 gave the Same results as SVM model 1,
# perhaps due to chance or choice of kernel given the structure of the data.
pred_train.3 <- predict(svp.3,train.data.1)
mean(pred_train.3==train.data.1$sim_death_calc)
pred_test.3 <- predict(svp.3,test.data.1)
mean(pred_test.3==test.data.1$sim_death_calc)

## This result tells us that the "tanhdot" kernel is not appropriate for this data:
# The fitted SVM model sucks, and the model's predictive power sucks.
# Possibly we could horse around with parameters and use a grid-search, but not today.
pred_train.4 <- predict(svp.4,train.data.1)
mean(pred_train.4==train.data.1$sim_death_calc)
pred_test.4 <- predict(svp.4,test.data.1)
mean(pred_test.4==test.data.1$sim_death_calc)

## SVM model 5, with the "laplacedot" kernel, is the clear winner:
pred_train.5 <- predict(svp.5,train.data.1)
mean(pred_train.5==train.data.1$sim_death_calc)
pred_test.5 <- predict(svp.5,test.data.1)
mean(pred_test.5==test.data.1$sim_death_calc)

## However, now we ask ourselves: if the target variable classes were approximately even,
# would this produce a better model fit? Keep in mind that there are many times more alive patients
# than dead patients in this dataset. Let's try balancing the target variable's classes with package UBL:

# Just using random over-sampling of the least occuring class of the target variable, which is dead patients.
library(UBL)
new_fake_cardio_data_1 <- UBL::RandOverClassif(sim_death_calc~.,cardio_nn_dataset)

dim(new_fake_cardio_data_1)
str(new_fake_cardio_data_1)

## For reference compare the original to the new target class balanced data:
# Notice we have waaaaay more dead people now, this ought to help the model fit improve.
table(cardio_nn_dataset$sim_death_calc)
table(new_fake_cardio_data_1$sim_death_calc)

## Make the new training and test dataset subsets from the new over-sampled dataset:
## Set seed to reproduce results obtained later on:
set.seed(1234)

## Use caret's native partition function to split the dataset:
training.sample.2 <- new_fake_cardio_data_1$sim_death_calc %>% 
  createDataPartition(p = 0.80, list = FALSE)
train.data.2  <- new_fake_cardio_data_1[training.sample.2, ]
test.data.2 <- new_fake_cardio_data_1[-training.sample.2, ]

## Give levels of the target variable names so "caret" doesn't get confused during nn computations.
levels(train.data.2$sim_death_calc) <- c("alive", "dead")
levels(test.data.2$sim_death_calc) <- c("alive", "dead")

head(train.data.2)
head(test.data.2)

## Finally, we train an SVM now on the top model from our previous smorgasbord:
svp.5_redox <- ksvm(sim_death_calc ~ ., data = train.data.2, type = "nu-svc", C = 1,
              kernel = "laplacedot", prob.model=TRUE, cross=10)

pred_train.5_redox <- predict(svp.5_redox,train.data.2)
mean(pred_train.5_redox==train.data.2$sim_death_calc)
pred_test.5_redox <- predict(svp.5_redox,test.data.2)
mean(pred_test.5_redox==test.data.2$sim_death_calc)
## Accuracy has actually dropped, but higher Kappa means the model
# is a slightly better fit to the data with this enlarged synthetic dataset.

## The best SVM "kernel" and type ("c-svc" vs "nu-svc") will
# depend on the structure and spread of your data.
# Now go forth and model!

