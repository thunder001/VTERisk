
message(normalizePath("~"))
rm(list = ls())
library(fst)
library(dplyr)
library(stringr)
library(plyr)
library(data.table)
library(caret)
library("Metrics")



setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
library('CalibrationCurves')

test3 = read.csv('M3_test_12_12.csv')
test4 = read.csv('M4_test_12_12.csv')

logit = function(p){  log(p /(1-p))}
expit = function(p){  exp(p)/ (1+exp(p)) }
 
setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/Cali_25_1212/Cali_Mats//')

# M1, M2 are 12/15a
M1 = read.csv('M1_Cali_Beta.csv')
M2 = read.csv('M2_Cali_Beta.csv')

# M3, M4 are for 12/12
M3 = read.csv('M3_Cali_Normal.csv')
M4 = read.csv('M4_Cali_Normal.csv')

M3 = data.frame( t(M3))
M4 = data.frame( t(M4))
m3 = M3[-1,]  
m4 = M4[-1,]


P3 = data.frame(p = rowMeans(m3), y = ifelse(test3$golds=="True", 1, 0)  )
P4 = data.frame(p = rowMeans(m4), y = ifelse(test4$golds=="True", 1, 0))
sum(P3$y)
sum(P4$y)

library(predtools)

calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred_T4_2, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))















Pred_T3_1 = make_subsample_spline(devs$Y3, tests$Y3)
Pred_T3_2 = make_subsample_spline(devs$Y3, tests$Y3)
calibration_plot(data = Pred_T3_1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred_T3_2, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))


Pred_T2_1 = make_subsample_spline(devs$Y2, tests$Y2)
Pred_T2_2 = make_subsample_spline(devs$Y2, tests$Y2)
calibration_plot(data = Pred_T2_1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred_T2_2, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))

Pred_T1_1 = make_subsample_spline(devs$Y1, tests$Y1)
Pred_T1_2 = make_subsample_spline(devs$Y1, tests$Y1)
calibration_plot(data = Pred_T1_1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))
calibration_plot(data = Pred_T1_2, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))

 
calibration_plot(data = Pred_T3_1, obs = 'y', pred = 'p', x_lim = c(0,.1), y_lim = c(0,.1))

Cali_1 = calibration_plot(data = Pred_T4_1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))


Pred = Pred_T4_1

?scale
?calibration_plot
Normalize_Cali = function(Pred){
  Pred$norm_p = Pred$p / max(Pred$p)
  calibration_plot(data = Pred, obs = 'y', pred = 'norm_p', x_lim = c(0,1), y_lim = c(0,1))
  
  Calib_Plot =  
}
 

 confusion_part = function(){

    
    
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
 }

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
