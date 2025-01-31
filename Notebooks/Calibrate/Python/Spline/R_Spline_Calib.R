
message(normalizePath("~"))
rm(list = ls())
library(fst)
library(dplyr)
library(stringr)
library(plyr)
library(data.table)
library(caret)
library("Metrics")

setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/CSV_preds/')



D = read.csv('devL_12_12.csv')
test = read.csv('testL_12_12.csv')

install.packages('CalibrationCurves')
library('CalibrationCurves')
genCalCurve()




split = function(MD){
  
  MD = MD[order(MD$patient_id),] 
  MD$X = MD$patient_id
  MD$pids = NULL
  MD$patient_id = NULL
  MD$dob = NULL
  MD$days_from_outcome = (as.Date(MD$outcome_date) - as.Date(MD$date)) %>% as.numeric()
  #MD$diag_date = NULL
  MD$daysfrom_index = (as.Date(MD$date) - as.Date(MD$index_date)) %>% as.numeric()
  MD$y =  ifelse(MD$golds == "True", 1, 0)
  MD$p = MD$probs #/ max(MD$probs)
  MD$censor_times = NULL
  
  MD$Diag_to_Index =(MD$index_date%>%as.Date() - MD$diag_date%>%as.Date())%>% as.numeric()
  X = select(MD,  pid =X,  "outcome_date" ,   "index_date",   "diag_date"  ,"outcome" ,  "y"  ,predicted_prob="probs"    ,  
             model_date ="dates",  "days_from_outcome",  "daysfrom_index"   )
  
  
  MD$l = logit(MD$p)
  
  firsts = !duplicated(MD$X)
  Y1 = MD[firsts,]
  Y_not1 = MD[!firsts,]
  
  seconds = !duplicated(Y_not1$X)
  Y2 = Y_not1[seconds,]
  Y_not2 = Y_not1[!seconds,]
  
  thirds = !duplicated(Y_not2$X)
  Y3 = Y_not2[thirds,]
  Y_not3 = Y_not2[!thirds,]
  
  fourths = !duplicated(Y_not3$X)
  Y4 = Y_not3[fourths,]
  
  
  y1 = select(Y1, X, p, l,  y )
  
  return(list(Y1=select(Y1, X, p, l,  y ), Y2=select(Y2, X, p, l,  y ), 
              Y3=select(Y3, X, p, l,  y ), Y4=select(Y4, X, p, l,  y ), 
              Y=select(MD, X, p, l,  y )))
  
}

devs = split(D)
tests = split(test)

hist(devs$Y1$p)
hist(devs$Y4$l)

library('betacal')
library('enmSdmX')



y1 = devs$Y1$y

dat = devs$Y1
hist(dat$l)
dat$scaled_logit = scale(dat$l)

hist(dat$scaled_logit)



NS = trainNS(data = dat, resp = 'y', preds = 'scaled_logit')

test_dat = tests$Y1
test_dat$scaled_logit = scale(test_dat$l)
pred = predictEnmSdm(NS, test_dat)

hist( pred )
libr



## Fitting beta calibration with three parameters

hist(p1)


max( devs$Y$p)

calib04 =  beta_calibration(devs$Y4$p  , devs$Y4$y, "abm")
beta_04 = beta_predict(devs$Y4$p,  calib04)

hist( beta_04)


calib1 <- beta_calibration(p1, y1, "abm")

beta_01 = beta_predict(p1,  calib1)

beta_11 = beta_predict(tests$Y1$p,  calib1)

hist(beta_11)






train1 = glm(  y ~l, family = 'binomial', data = devs$Y1)
train4 = glm(  y ~l, family = 'binomial', data = devs$Y4)

summary(train1)
summary(train4)

library(glmnet)

x = as.matrix(cbind(1, devs$Y4$l))

train4 = cv.glmnet( x = x,
                    y =y, 
                    family = 'binomial',type.measure = 'auc')

summary(train4)
#Logit4= 


pHat = predict(train4, newx = cbind(1, tests$Y4$l),
               type = 'response')

hist(pHat)

pHat = predict(train4, newdata = (tests$Y4), type = 'response', 
               )

hist( pHat)
calPerf = val.prob.ci.2(pHat, tests$Y4$y)
 









confu = function(p, yy, THRESH=.5){
  CM = confusionMatrix( ifelse(p > THRESH,1,0) %>% as.factor(), yy %>% as.factor())
  t(CM$table)
}


pROC::auc( MD$y,  (MD$probs) )
confu (  (MD$probs) , MD$y)

mean(Y1$probs) 
mean(Y1$y) 

pROC::auc( Y1$y,  (Y1$probs) )
pROC::auc( Y2$y,  (Y2$probs) )
pROC::auc( Y3$y,  (Y3$probs) )
pROC::auc( Y4$y,  (Y4$probs) )
 
confu (Y1$probs, Y1$y, .2506)
confu (Y2$probs, Y2$y, .1720)
confu (Y3$probs, Y3$y, .1249) 
confu (Y4$probs, Y4$y, .1137)
confu (Y4$probs, Y4$y, .105)


 
 
 cor(Y1$exams, Y1$probs)
 cor(Y2$exams, Y2$probs)
 cor(Y3$exams, Y3$probs)
 cor(Y4$exams, Y4$probs)

pos_analysis = function(){
  
  V1 = Y1[Y1$y==1,]
  V2= Y2[Y2$y==1,]
  V3= Y3[Y3$y==1,]
  V4= Y4[Y4$y==1,]
  
  900 + 368 + 208 + 145
  796 + 390+194+131

  hist(V1$exams)
  
  
  cor(V1$exams, V1$probs)
  cor(V2$exams, V2$probs)
  cor(V3$exams, V3$probs)
  cor(V4$exams, V4$probs)
  
  
  hist(V1$exams)
  hist(V2$exams)
  hist(V3$exams)
  hist(V4$exams)
  
  
  hist(Y1$exams)
  hist(Y2$exams)
  hist(Y3$exams)
  hist(Y4$exams)
    
  hist(V1$exams[V1$exams<1000])
  hist(V2$exams)
  hist(V2$exams)
  hist(V4$exams)
  
  summary(V1$exams)
  summary(V2$exams)
  
}
