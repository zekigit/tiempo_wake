library(ggplot2)
library(tidyr)
library(dplyr)
library(popbio)
library(pscl)

study_path = '/Volumes/FAT/Time/ETG_scalp/'

# Load Time-Frequency
tf_raw = read.csv(file.path(study_path, 'tables', 's_trial_dat.csv'))
tf_raw$Diff[tf_raw$Order==1] = tf_raw$Comparison[tf_raw$Order==1] - tf_raw$Standard [tf_raw$Order==1]
tf_raw$Diff[tf_raw$Order==2] = tf_raw$Standard[tf_raw$Order==2] - tf_raw$Comparison [tf_raw$Order==2]
tf_raw$type = factor(tf_raw$condition, levels = c(90, 70), labels = c('lon', 'sho'))

tf_ok <- tf_raw %>% gather('roi', 'power', 18:23)

ggplot(tf_ok[tf_ok$roi == 'f',], aes(roi, power, fill=type)) + geom_boxplot()

tf_ok$type_bin[tf_raw$type == 'lon'] = 1
tf_ok$type_bin[tf_raw$type == 'sho'] = 0

# Results by ROI
for (roi in (c('a', 'b', 'c', 'd', 'e', 'f'))) {
print(roi)
roi_dat = tf_ok[tf_ok$roi == roi, ]
power = roi_dat$power
condition = roi_dat$type_bin
#condition = roi_dat$Accuracy

logi.hist.plot(power, condition, boxp = FALSE, type = "hist", col = "grey", xlabel =  roi,
               ylabel = "Condition")

bin_reg <- glm(roi_dat$type_bin ~ roi_dat$power, family = binomial())
print(summary(bin_reg))
print(pR2(bin_reg))

# l_reg <- lm(roi_dat$RT ~ roi_dat$power)
# print(summary(l_reg))
}

ggplot(tf_ok, aes(x= RT, y=power)) + geom_point(shape=1) + geom_smooth(method=lm) + xlab('RT') + facet_wrap(~roi)


