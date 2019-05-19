## Simulate some realistic, but fake, ra fibrosis datasets to see if 
# we can predict UT stage fibrosis score with such a small data set.
# Hoffmann et al. (2019) proposed simulating additional data for datasets
# to augment machine learning as one solution to "the curse of small sample sizes."
# and/or expensive to collect data.

## For the original dataset N=209. The simulated datasets here are larger (N=500),
# but are drawn from identical distributions and
# have the same mean, sd and lower and upper bounds
# in the case of the numeric variables.

library(msm)
# used to draw from truncated distribution;
# bounded by the max, min, sd and mean of the real data.
# The real data was drawn from a large University of Utah database months ago.

  # Simulate some male and female study subjects.:
sim_gender<- rbinom(1, n = 500, p = 0.5)
  
  # Simulate some binary comorbidities the test subjects may or may not suffer from.:
sim_diab <- rbinom(1, n = 500, p = 0.5)
sim_sleep_ap <<- rbinom(1, n = 500, p = 0.5)
sim_hypertension <<- rbinom(1, n = 500, p = 0.5)
sim_stroke <<- rbinom(1, n = 500, p = 0.5)
sim_ablate <<- rbinom(1, n = 500, p = 0.5)

  # Simulate draws from truncated normal distributions for numerical variables.:
sim_age <- rtnorm(n = 500, mean = 65.3, sd = 16.69, lower = 21, upper = 93) # age
sim_bmi <- rtnorm(n = 500, mean = 27.74, sd = 5.313, lower = 17.33, upper = 46.52) # bmi
sim_ra_fibrosis <- rtnorm(n = 500, mean = 4.9435, sd = 4.9435, lower = 0, upper = 27.1) # ra fibrosis


  ## The Utah Stage Fibrosis Score ranges from 1-3 in the real cardiac dataset:
# 1 isn't too bad, a little fibrotic tissue, but you might be able to wait to have an ablation.
# 2 means you probably need an ablation soon. 
# 3 means you probably need an ablation immediately, your heart is pretty much toast.
utah_score <- rbinom(2, n = 500, p = 1/3)

  ## Bind all the simulated vectors together.:
sim_quants <- cbind(sim_gender, sim_age, sim_bmi, sim_diab, sim_hypertension, sim_sleep_ap, sim_stroke, sim_ablate,
                    sim_ra_fibrosis, utah_score)

  ## Save your dataframe.:
simulated_fibrosis_data <- as.data.frame(sim_quants)
simulated_fibrosis_data$sim_gender <- as.factor(simulated_fibrosis_data$sim_gender)

  ## Make gender a proper factor:
levels(simulated_fibrosis_data$sim_gender) <- c("Female","Male")

  ## Give the simulated dataset a nice name:
fake_ra_fibrosis_data_1 <- simulated_fibrosis_data
write.csv(fake_ra_fibrosis_data_1,"fake_ra_fibrosis_data_1.csv")

# There you go! Have fun making your own fake data.
# And, yeah, one can probably say that the mean and sd
# should be drawn from Bootstrapped Confidence Intervals instead,
# but we can talk about that another time.
# This data we just made is actually indistinguishable from the real data.
