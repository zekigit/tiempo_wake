library(ggplot2)
library(tidyr)
library(dplyr)
library(popbio)

study_path = '/Users/lpen/Documents/Experimentos/Drowsy Time/TimeGeneralization/ieeg/'

# Load Time-Frequency
tf_raw = read.csv(file.path(study_path, 'tables', 'log_and_power_dat.csv'))
tf_raw = na.omit(tf_raw)
tf_raw$Diff[tf_raw$Order==1] = tf_raw$Comparison[tf_raw$Order==1] - tf_raw$Standard [tf_raw$Order==1]
tf_raw$Diff[tf_raw$Order==2] = tf_raw$Standard[tf_raw$Order==2] - tf_raw$Comparison [tf_raw$Order==2]
#tf_raw = tf_raw[tf_raw$Accuracy == 1, ]
#tf_raw = tf_raw[tf_raw$RT > 0.4, ]
#tf_raw = tf_raw[tf_raw$Diff > 600, ]


ggplot(tf_raw[tf_raw$HP3 < 20,], aes(s2, HP3)) + geom_violin()
ggplot(tf_raw, aes(Diff, HP3)) + geom_point()


longer = tf_raw[tf_raw$s2 == 'longer', ]
shorter = tf_raw[tf_raw$s2 == 'shorter', ]

sd_out = 3
chans = c('SC4', 'HP3', 'HPi3')
chans = c('HP3')

df_list <- list(longer, shorter)
lapply(df_list, function(x) {
  
  for (ch in chans) {
  # Delete power outliers
  dat_ok <- x[x[[ch]] > mean(x[[ch]]) - sd_out * sd(x[[ch]]) & x[[ch]] < mean(x[[ch]]) + sd_out * sd(x[[ch]]), ]
    
  # Delete RT outliers
  dat_ok <- dat_ok[dat_ok$RT > mean(dat_ok$RT) - sd_out * sd(dat_ok$RT) 
                                       & dat_ok$RT < mean(dat_ok$RT) + sd_out * sd(dat_ok$RT), ]
  
  response = dat_ok$Accuracy
  pow = dat_ok[[ch]]
  logi.hist.plot(pow, response, boxp = FALSE, type = "hist", col = "grey", xlabel =  ch,
                 ylabel = "Response")

  RT = dat_ok$RT
  pow = dat_ok[[ch]]
  
  rt_mod = lm(RT ~ pow)
  print(ch)
  print(summary(rt_mod))

  print(ggplot(dat_ok, aes_string(x= RT, y=ch)) + geom_point(shape=1) + geom_smooth(method=lm) + xlab('RT'))
  }
})


l_reg <- glm(tf_raw$s2 ~ tf_raw$HP3, family = binomial())
summary(l_reg)

tf_raw$type[tf_raw$s2 == 'longer'] = 1
tf_raw$type[tf_raw$s2 == 'shorter'] = 0

power = tf_raw$HP3
condition = tf_raw$type

logi.hist.plot(power, condition, boxp = FALSE, type = "hist", col = "grey", xlabel =  "gamma-H", 
               ylabel = "Condition")
