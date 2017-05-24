library(ggplot2)
library(tidyr)
library(dplyr)
library(R.matlab)

study_path = '/Volumes/FAT/Time/ETG_scalp/'

# Load Time-Frequency
tf_raw = read.csv(file.path(study_path, 'tables', 'power_data.csv'))
tf_raw$subject = factor(tf_raw$subject)

#load behaviour
behav_data_base <- readMat(file.path(study_path, 'tables','ETG_scalp_Data_n22.mat'))

behav_data_raw <- data.frame(behav_data_base$DataOK)
variable_names<-c("Event_number", "Block_number", "Duration", "C_Ratio", "Standard", "Comparison", 
                  "Gap", "Event_order", "RT", "Response", "Accuracy", "Proportion","Second_Difference", 
                  "C_Difference", "Subject")
names(behav_data_raw) <-make.names(variable_names)  

# Make Duration a factor and assign names
behav_data_raw$Duration=factor(behav_data_raw$Duration, levels = c(1,2), labels = c("Sub-second", "Supra-second"))
behav_data_raw$Event_order=factor(behav_data_raw$Event_order, levels = c(1,2), labels = c("S-C", "C-S"))
# Keep good subjects
behav_ok<-na.omit(behav_data_raw)
behav_ok <- behav_ok[which(behav_ok$Subject != 11 & behav_ok$Subject != 19),]
behav_ok$Response[behav_ok$Subject == 17] = !behav_ok$Response[behav_ok$Subject == 17]
# Keep supra-second condition
behav_ok = behav_ok[behav_ok$Duration == 'Supra-second',]

acc = aggregate(behav_ok['Accuracy'], list(behav_ok$Subject), mean)
colnames(acc)[1] <- 'subject'
acc$subject = factor(acc$subject)
hist(acc$Accuracy)

all_dat <- inner_join(tf_raw, acc, by='subject')
lon_dat <- all_dat[all_dat$condition == 'longer',]

plt_cor <- ggplot(lon_dat, aes(Accuracy, power)) + geom_point() + geom_smooth(method = "lm") + 
  facet_wrap(~roi)

ggplot(tf_raw, aes(roi, power, fill=condition)) + geom_violin()

