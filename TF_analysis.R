library(ggplot2)
library(tidyr)

study_path = '/Volumes/FAT/Time/ETG_scalp/'
dat_raw = read.csv(file.path(study_path, 'tables', 'power_data_B2.csv'))

dat_ok <- dat_raw %>% gather('Cond', 'Zscore', 2:3) 
ggplot(dat_ok, aes(Cond, Zscore, fill=Cond)) + geom_violin()
