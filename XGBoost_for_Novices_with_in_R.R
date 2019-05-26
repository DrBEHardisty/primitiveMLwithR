## Can we predict which patients are alive and which are dead
# with high accuracy with eXtreme Gradient Boosting?
# We'll use 2 synthetic datasets, one very realistic,
# and one made of repeated random samples of over- or under- represented classes of our targe variable.
# (That is, the variable who's binary outcomes we're looking to predict.)

## Load Worthington's "ipak" function:
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

## Give "packages" the names of the packages you'll use in your data tasks:
packages <- c("tidyverse","dplyr","magrittr","skimr","ggplot2","caret","janitor","msm","UBL","xgboost")

## Run "ipak" and open all the packages:
ipak(packages)

## For the original dataset N = 19,197.
# The simulated dataset here is exactly the same size,
# Some binomial probs. are biased by how the real data were biased.
# The original (real) data were drawn from a University of Utah Div. of Cardiology database months ago.

# Simulate some male and female study subjects.:
sim_gender <- rbinom(1, n = 19197, p = 0.58)

# Simulate some binary comorbidities the test subjects may or may not suffer from.:
sim_drugs <- rbinom(1, n = 19197, p = 0.04)
sim_alcoholism <- rbinom(1, n = 19197, p = 0.05)
sim_anemia <- rbinom(1, n = 19197, p = 0.26)
sim_bloodloss <- rbinom(1, n = 19197, p = 0.031)
sim_obesity <- rbinom(1, n = 19197, p = 0.22)
sim_coagulopathy <- rbinom(1, n = 19197, p = 0.12)
sim_rheumatic <- rbinom(1, n = 19197, p = 0.10)
sim_tumor <- rbinom(1, n = 19197, p = 0.07)
sim_mets <- rbinom(1, n = 19197, p = 0.05)
sim_lymphoma <- rbinom(1, n = 19197, p = 0.02)
sim_hiv <- rbinom(1, n = 19197, p = 0.004)
sim_pud <- rbinom(1, n = 19197, p = 0.26)
sim_liver <- rbinom(1, n = 19197, p = 0.08)
sim_renal <- rbinom(1, n = 19197, p = 0.20)
sim_hypothyroid <- rbinom(1, n = 19197, p = 0.20)
sim_diab <- rbinom(1, n = 19197, p = 0.07)
sim_pulmonary <- rbinom(1, n = 19197, p = 0.23)
sim_neuroother <- rbinom(1, n = 19197, p = 0.29)
sim_paralysis <- rbinom(1, n = 19197, p = 0.05)
sim_htn <- rbinom(1, n = 19197, p = 0.5)
sim_pvd <- rbinom(1, n = 19197, p = 0.20)
sim_phtn <- rbinom(1, n = 19197, p = 0.06)
sim_valvular  <- rbinom(1, n = 19197, p = 0.35)
sim_chf <- rbinom(1, n = 19197, p = 0.32)
sim_death_calc <- rbinom(1, n = 19197, p = 0.12)

## Bind all the simulated vectors together.:
sim_quants <- cbind(sim_gender,sim_drugs,sim_alcoholism,sim_anemia,sim_bloodloss,sim_obesity,sim_coagulopathy,sim_rheumatic,
                    sim_tumor,sim_mets,sim_lymphoma,sim_hiv,sim_pud,sim_liver,sim_renal,sim_hypothyroid,sim_diab,sim_pulmonary,
                    sim_neuroother,sim_paralysis,sim_htn,sim_pvd,sim_phtn,sim_valvular,sim_chf,sim_death_calc)

## Save your dataframe.:
simulated_cardio_death_data <- as.data.frame(sim_quants)
simulated_cardio_death_data$sim_gender <- as.factor(simulated_cardio_death_data$sim_gender)

## Make gender a proper factor:
levels(simulated_cardio_death_data$sim_gender) <- c("Female","Male")

## Give the simulated dataset a nice name:
fake_cardio_death_data_1 <- simulated_cardio_death_data

## Save your new fake dataset:
write.csv(fake_cardio_death_data_1,"fake_cardio_death_data_1.csv")

# Read in the simulated data, maybe save on memory?:
cardio_nn_dataset <- read.csv("fake_cardio_death_data_1.csv")

## Use R package "janitor" to "clean" column names.
# (Some ML packages get glitchy over R column names).
cardio_nn_dataset <- cardio_nn_dataset %>% clean_names()

## If these give different names, some columns need to be renamed for ease of handling by R package "caret".
names(cardio_nn_dataset)
make.names(names(cardio_nn_dataset))

dim(cardio_nn_dataset)
str(cardio_nn_dataset)

## Make sure all the factors are factors:
shouldBeFactor <- c("sim_drugs","sim_alcoholism","sim_alcoholism","sim_anemia","sim_bloodloss","sim_obesity",
                    "sim_coagulopathy","sim_rheumatic","sim_tumor","sim_mets","sim_lymphoma","sim_hiv","sim_pud",
                    "sim_liver","sim_renal","sim_hypothyroid","sim_diab","sim_pulmonary","sim_neuroother",
                    "sim_paralysis","sim_htn","sim_pvd","sim_phtn","sim_valvular","sim_chf","sim_death_calc")

for(v in shouldBeFactor) {
  cardio_nn_dataset[[v]] <- as.factor(cardio_nn_dataset[[v]])
}

glimpse(cardio_nn_dataset)

#########
## Now we can fit an xg boost model on subsets of the cardiology dataset.
# Then we test the training model on a test subset.
library(data.table)

## Set seed for reproducility:
set.seed(34)
training.sample.1 <- cardio_nn_dataset$sim_death_calc %>% 
  createDataPartition(p = 0.80, list = FALSE)
train.data.1  <- cardio_nn_dataset[training.sample.1, ]
test.data.1 <- cardio_nn_dataset[-training.sample.1, ]

## Get rid of the column called "x":
train.data.1 <- train.data.1 %>% select(-x)
test.data.1 <- test.data.1 %>% select(-x)

## View outputs of column removal:
names(train.data.1)
names(test.data.1)

head(train.data.1)
head(test.data.1)

## Construct your R package "xgboost" data objects:
xgdata.1 <- data.table(train.data.1,keep.rownames = FALSE)
dim(xgdata.1)
str(xgdata.1)

levels(xgdata.1[,sim_death_calc])

sparse_xgbmatrix.1 <- model.matrix(sim_death_calc ~ ., data = xgdata.1)[,-1]
head(sparse_xgbmatrix.1)

output_vector.1 = xgdata.1[,sim_death_calc] == "1"

xgdata.2 <- data.table(test.data.1,keep.rownames = FALSE)
dim(xgdata.2)
str(xgdata.2)

levels(xgdata.2[,sim_death_calc])

sparse_xgbmatrix.2 <- model.matrix(sim_death_calc ~ ., data = xgdata.2)[,-1]
head(sparse_xgbmatrix.2)

output_vector.2 = xgdata.2[,sim_death_calc] == "1"

bst <- xgboost(data = sparse_xgbmatrix.1,
               label = output_vector.1,
               max_depth = 15,
               eta = 1,
               nrounds = 100,
               objective = "binary:logistic")

print(bst)

## See which features are featured in every final xg boost model:
importance.1 <- xgb.importance(feature_names = colnames(sparse_xgbmatrix.1), model = bst)
head(importance.1)

importanceRaw <- xgb.importance(feature_names = colnames(sparse_xgbmatrix.1), model = bst)
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]
print(importanceClean)

## Plot the important variables:
xgb.plot.importance(importance_matrix = importance.1)

## Now make model predictions on the test data:
xgbpred <- predict(bst,newdata = sparse_xgbmatrix.2)
xgbpred <- ifelse(xgbpred > 0.5,1,0)

## Make a confusion matrix:
confusionMatrix(as.factor(xgbpred),test.data.1[,26])

## Accuracy = 85.2%, but Kappa = 0.03, which means the model doesn't
# fit the real data values particularly well. It's a poor fit.

# Now, we'll fit XG Boost models to a synthetic dataset created by "UBL".
# Just using random over-sampling of the least occuring class of the target variable from our cardio_nn_dataset.
new_fake_cardio_data_1 <- UBL::RandOverClassif(sim_death_calc~.,cardio_nn_dataset)

dim(new_fake_cardio_data_1)
str(new_fake_cardio_data_1)

# Partition into training and test data
# Set seed for reproducibility.
set.seed(47)
index <- createDataPartition(new_fake_cardio_data_1$sim_death_calc, p = 0.7, list = FALSE)
train_data <- new_fake_cardio_data_1[index, ]
test_data  <- new_fake_cardio_data_1[-index, ]

## Get rid of variable column "x":
train_data <- train_data %>% select(-x)
test_data <- test_data %>% select(-x)

## Inspect output:
names(train_data)
names(test_data)

## Now, as we did above, make 2 xg boost datasets for 
# model training and model testing out of our train and test datasets.

xgdata.1 <- data.table(train_data,keep.rownames = FALSE)
dim(xgdata.1)
str(xgdata.1)

levels(xgdata.1[,sim_death_calc])

sparse_xgbmatrix.1 <- model.matrix(sim_death_calc ~ ., data = xgdata.1)[,-1]
head(sparse_xgbmatrix.1)

output_vector.1 = xgdata.1[,sim_death_calc] == "1"

xgdata.2 <- data.table(test_data,keep.rownames = FALSE)
dim(xgdata.2)
str(xgdata.2)

levels(xgdata.2[,sim_death_calc])

sparse_xgbmatrix.2 <- model.matrix(sim_death_calc ~ ., data = xgdata.2)[,-1]
head(sparse_xgbmatrix.2)

output_vector.2 = xgdata.2[,sim_death_calc] == "1"

bst.2 <- xgboost(data = sparse_xgbmatrix.1,
               label = output_vector.1,
               max_depth = 15,
               eta = 1,
               nrounds = 100,
               objective = "binary:logistic")

print(bst.2)

## See which features are featured in every final xg boost model:
importance.1 <- xgb.importance(feature_names = colnames(sparse_xgbmatrix.1), model = bst.2)
head(importance.1)

importanceRaw <- xgb.importance(feature_names = colnames(sparse_xgbmatrix.1), model = bst.2)
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]
print(importanceClean)

## Plot the important variables:
xgb.plot.importance(importance_matrix = importance.1)

## Make predictions on the test data:
xgbpred <- predict(bst.2,newdata = sparse_xgbmatrix.2)
xgbpred <- ifelse(xgbpred > 0.5,1,0)

## Make a confusion matrix:
confusionMatrix(as.factor(xgbpred),test_data[,26])

## Accuracy is 80.5% and Kappa = 0.61, that's a good fit of the model to the real data values.