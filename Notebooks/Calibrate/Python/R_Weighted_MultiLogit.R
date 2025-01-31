
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


library(predtools)
library('CalibrationCurves')

logit = function(p){  log(p /(1-p) )}
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
 

## calculate wieghts from multi

wt1 = ifelse(devs$Y1$y ==1, 1/mean(devs$Y1$y), 1)
wt2 = ifelse(devs$Y2$y ==1, 1/mean(devs$Y2$y), 1)
wt3 = ifelse(devs$Y3$y ==1, 1/mean(devs$Y3$y), 1)
wt4 = ifelse(devs$Y4$y ==1, 1/mean(devs$Y4$y), 1)

wt1 = ifelse(devs$Y1$y ==1, 1/  ( mean(devs$Y1$y) * .064 ), 1)
wt2 = ifelse(devs$Y2$y ==1, 1/ ( mean(devs$Y2$y) * .064 ), 1)
wt3 = ifelse(devs$Y3$y ==1, 1/ ( mean(devs$Y3$y) * .064 ), 1)
wt4 = ifelse(devs$Y4$y ==1, 1/ ( mean(devs$Y4$y) * .064 ), 1)


joining_weights = function(){
    
    Y12 = join( select( devs$Y1, X, y1=y) , select( devs$Y2, X, y2 =y ) , by = 'X')
    Y123 = join( Y12, select( devs$Y3, X, y3=y), by = 'X')
    Z = join( Y123, select( devs$Y4, X, y4=y), by = 'X')
    
    Z$w1 = ifelse(Z$y1 ==1, 1/mean(Z$y1), 1)
    Z$w2 = ifelse(Z$y2 ==1, 1/mean(Z$y2, na.rm = T), 1)
    Z$w3 = ifelse(Z$y3 ==1, 1/mean(Z$y3, na.rm = T), 1)
    Z$w4 = ifelse(Z$y4 ==1, 1/mean(Z$y4, na.rm = T), 1)
    
    # 1010432232 is one where occurs at 4th
    Z$w1 [Z$y4 ==1] = 0
    Z$w2 [Z$y4 ==1] =0
    Z$w3 [Z$y4 ==1] =0
    #Z$w4 [Z$y4 ==1] = Z$w4 [Z$y4 ==1]) /mean(any_VTE)
    
    Z$w1 [Z$y3 ==1] =0
    Z$w2 [Z$y3 ==1] = 0
    #Z$w3 [Z$y3 ==1] = Z$w3 [Z$y3 ==1]/mean(any_VTE)
    
    Z$w1 [Z$y2 ==1] = 0
    #Z$w2 [Z$y2 ==1] = Z$w2 [Z$y2 ==1]/mean(any_VTE)
    
    n_present = rowSums(!is.na(select(Z, y1, y2, y3, y4)))
    
    #Z$y2[is.na(Z$y2)] = -1
    #Z$y3[is.na(Z$y3)] = -1
    #Z$y4[is.na(Z$y4)] = -1
    
    #Z$w1[Z$y1 ==0] =( Z$w1 / n_present ) [Z$y1 ==0]
    #Z$w2[Z$y2 ==0] = (Z$w2 /n_present ) [Z$y2 ==0]
    #Z$w3[Z$y3 ==0] = (Z$w3 /n_present) [Z$y3 ==0]
    #Z$w4[Z$y4 ==0] = (Z$w4 /n_present) [Z$y4 ==0]
    
    ww1 =   select (  join(devs$Y1, Z)  , w1)
    ww2 =   select (  join(devs$Y2, Z)  , w2)
    ww3 =   select (  join(devs$Y3, Z)  , w3)
    ww4 =   select (  join(devs$Y4, Z)  , w4) 
}
train1_cv = cv.glmnet(x = cbind(1,devs$Y1$l ) , y =devs$Y1$y , weights = (ww1$w1))
train2_cv = cv.glmnet(x = cbind(1,devs$Y2$l ) , y =devs$Y2$y , weights = (ww2$w2))
train3_cv = cv.glmnet(x = cbind(1,devs$Y3$l ) , y =devs$Y3$y , weights = (ww3$w3))
train4_cv = cv.glmnet(x = cbind(1,devs$Y4$l ) , y =devs$Y4$y , weights = (ww4$w4))

train1_cv = cv.glmnet(x = cbind(1,devs$Y1$l ) , y =devs$Y1$y , weights = (wt1))
train2_cv = cv.glmnet(x = cbind(1,devs$Y2$l ) , y =devs$Y2$y , weights = (wt2))
train3_cv = cv.glmnet(x = cbind(1,devs$Y3$l ) , y =devs$Y3$y , weights = (wt3))
train4_cv = cv.glmnet(x = cbind(1,devs$Y4$l ) , y =devs$Y4$y , weights = (wt4))

cv_pred1 = predict(train1_cv, newx = cbind(1, tests$Y1$l), type = 'response')
cv_pred2 = predict(train2_cv, newx = cbind(1, tests$Y2$l), type = 'response')
cv_pred3 = predict(train3_cv, newx = cbind(1, tests$Y3$l), type = 'response')
cv_pred4 = predict(train4_cv, newx = cbind(1, tests$Y4$l), type = 'response')

hist( cv_pred1 )
hist( cv_pred2 )
hist( cv_pred3 )
hist( cv_pred4 )

Pred1 = data.frame(p = cv_pred1[,1], y = tests$Y1$y)
Pred2 = data.frame(p = cv_pred2[,1], y = tests$Y2$y)
Pred3 = data.frame(p = cv_pred3[,1], y = tests$Y3$y)
Pred4 = data.frame(p = cv_pred4[,1], y = tests$Y4$y)



calibration_plot(data = Pred1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred2, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred3, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred4, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))

 confusion = function(){

   pROC::auc( tests$Y1$y,  tests$Y1$p )
   pROC::auc( tests$Y2$y,  tests$Y2$p )
   pROC::auc( tests$Y3$y,  tests$Y3$p )
   pROC::auc( tests$Y4$y,  tests$Y4$p )
   
   
   pROC::auc( tests$Y1$y,  Pred1$p )
   pROC::auc( tests$Y2$y,  Pred2$p )
   pROC::auc( tests$Y3$y,  Pred3$p )
   pROC::auc( tests$Y4$y,  Pred4$p )
   
    
    
    confu = function(yy, p, THRESH=.5){
      CM = confusionMatrix( ifelse(p > THRESH,1,0) %>% as.factor(), yy %>% as.factor());  t(CM$table)
    }
    
    confu( tests$Y1$y,  Pred1$p )
    confu( tests$Y2$y,  Pred2$p )
    confu( tests$Y3$y,  Pred3$p )
    confu( tests$Y4$y,  Pred4$p )
    
    
      
      
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


 }
 
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
