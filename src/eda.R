##################################################################
# downloading packages if anything is mmissing
##################################################################

list.of.packages <- c('data.table', 'stringr', 'dplyr', 'fasttime')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

##################################################################
# END: downloading packages if anything is mmissing
##################################################################

library('data.table')
library('stringr')
library('dplyr')
library('tidyr')
library('purrr')
library('stringr')
library('ggplot2')
library('fasttime')
library('lubridate')

rm(list=ls())

DATA_DIR = 'd:/ml/kaggle/competitive_data_science/data/'
setwd(DATA_DIR)
# some random location to look at the intermediate results
TEMP1 = str_c(DATA_DIR, 'temp1.csv')
TEMP2 = str_c(DATA_DIR, 'temp2.csv')
TEMP3 = str_c(DATA_DIR, 'temp3.csv')

data = data.table::fread('sales_train.csv.gz')
data[, date:=as.POSIXct(date, format='%d.%m.%Y', tz='GMT')]
data[, year:=year(date)]
data[, month:=month(date)]
# data[, `:=`(shop_id=as.factor(shop_id), item_id=as.factor(item_id))]


data_m = data[, .(item_cnt_month=sum(item_cnt_day), date=max(date), .N), by=c('year', 'month', 'shop_id', 'item_id')]
data_m[, date:=fastPOSIXct(as.Date(trunc.POSIXt(date, units='month')) + months(1) - 1, tz='GMT')]

# expanding the data with the rows for each (shop, item) since the product was first sold
all_dates = data_m$date %>% unique() %>% sort()
data_m2 = merge(
  data_m[, .(date=all_dates), by=c('shop_id', 'item_id')],
  data_m[, -c('year', 'month', 'N')],
  by=c('shop_id', 'item_id', 'date'), all=T)
data_m2[is.na(item_cnt_month), item_cnt_month:=0]

data_test = data.table::fread('test.csv.gz')

setkey(data_m2, shop_id, item_id, date)
data_m3 = data_m2[, .(item_cnt_month=tail(item_cnt_month, 1)), by=c('shop_id', 'item_id')]
data_m4 = merge(
  data_m3,
  data_test,
  by=c('shop_id', 'item_id'), all.y=T)
data_m4[is.na(item_cnt_month), item_cnt_month:=0]
data_m4[item_cnt_month > 20, item_cnt_month:=20]
data_m4[item_cnt_month <  0, item_cnt_month:= 0]

to_csv = function(dt, fname){
  data.table::fwrite(dt, file=fname, sep=',', dateTimeAs='write.csv')
}


data_m4[, .(ID, item_cnt_month)][order(ID)] %>% to_csv('submission_prev_month3.csv')

s1 = fread('submission_prev_month.csv')
s2 = fread('submission_prev_month2.csv')
s3 = fread('submission_prev_month3.csv')

merge(s1, s3, 'ID', all=T)[item_cnt_month.x!=item_cnt_month.y]
merge(s2, s3, 'ID', all=T)[item_cnt_month.x!=item_cnt_month.y]

setkey(data_m2, shop_id, item_id)
ll = data_test[, .(shop_id, item_id)] %>% as.list()

data_m2[data_test[, .(shop_id, item_id)] %>% as.list()]
setdiff(data_m2[k1, shop_id] %>% unique(), k1)
setdiff(k1, data_m2[k1, shop_id] %>% unique())


# items with > 0 median per month sales over past 12 months
data_good_items = data_m2[date %in% tail(all_dates, 1), .(item_cnt_month_median=median(item_cnt_month)), by='item_id'][item_cnt_month_median > 0][order(-item_cnt_month_median)]
good_items = data_good_items[, item_id]

data_m3 = data_m2[item_id %in% good_items]

data_total = data_m3[, .(item_cnt=sum(item_cnt_month), .N), by=c('shop_id')][order(-item_cnt)]
shop_ids = data_total[item_cnt %in% quantile(data_total$item_cnt, seq(0, 1, 0.1), type=1), shop_id]

data_total_item = data_m3[shop_id == shop_ids[5], .(item_cnt=sum(item_cnt_month)), by=c('item_id')][order(-item_cnt)]
item_ids = data_total_item[item_cnt %in% quantile(data_total_item$item_cnt, seq(0, 1, 0.1), type=1), .(item_id=item_id[1]), by='item_cnt']$item_id

data_m3[(shop_id == shop_ids[5]) & (item_id == item_ids[1])] %>% ggplot(aes(x=date, y=item_cnt_month)) + geom_point() + geom_line()
data_m3[(shop_id == shop_ids[5]) & (item_id == item_ids[4])] %>% ggplot(aes(x=date, y=item_cnt_month)) + geom_point() + geom_line()

data_m3[(item_id == item_ids[5]), .(item_cnt_month=sum(item_cnt_month)), by=c('item_id', 'date')] %>% ggplot(aes(x=date, y=item_cnt_month)) + geom_point() + geom_line()

