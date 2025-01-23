
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

test1 = read.csv('M1_test_12_12.csv')
test2 = read.csv('M2_test_12_12.csv')
test3 = read.csv('M3_test_12_12.csv')
test4 = read.csv('M4_test_12_12.csv')

logit = function(p){  log(p /(1-p))}
expit = function(p){  exp(p)/ (1+exp(p)) }


par(mfrow = c(1,4))
P1 = data.frame(pid = test1$patient_id, p = test1$probs, y = ifelse(test1$golds=="True", 1, 0)  )
P2 = data.frame(pid = test2$patient_id,p =  test2$probs, y = ifelse(test2$golds=="True", 1, 0)  )
P3 = data.frame(pid = test3$patient_id,p =  test3$probs, y = ifelse(test3$golds=="True", 1, 0)  )
P4 = data.frame(pid = test4$patient_id,p =  test4$probs, y = ifelse(test4$golds=="True", 1, 0)  )


test1 = read.csv('M1_test_12_12.csv')
test2 = read.csv('M2_test_12_12.csv')
test3 = read.csv('M3_test_12_12.csv')
test4 = read.csv('M4_test_12_12.csv')

calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
 
 
Cal1  = calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.35), y_lim = c(0,.35), xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)")
Cal2  = calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.35), y_lim = c(0,.35), xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")
Cal3 = calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.35), y_lim = c(0,.35), xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")
Cal4 = calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.35), y_lim = c(0,.35), xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")

 
G12 = grid.arrange(Cal1$calibration_plot, Cal2$calibration_plot, nrow =1)
G34 = grid.arrange(Cal3$calibration_plot, Cal4$calibration_plot, nrow = 1)

#grid.arrange(grobs = lapply(pl, "+", margin))

grid.arrange(G12, G34, nrow = 2)



Cal1  = calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =20)
Cal2  = calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)", nTiles =20)
Cal3 = calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)",nTiles =20)
Cal4 = calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)",nTiles =20)


G12 = grid.arrange(Cal1$calibration_plot, Cal2$calibration_plot, nrow =1)
G34 = grid.arrange(Cal3$calibration_plot, Cal4$calibration_plot, nrow = 1)

#grid.arrange(grobs = lapply(pl, "+", margin))

grid.arrange(G12, G34, nrow = 2)




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
