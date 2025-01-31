
message(normalizePath("~"))
rm(list = ls())
library(fst)
library(dplyr)
library(stringr)
library(plyr)
library(data.table)
library(caret)
library("Metrics")
library(egg)

rm(list = ls())

setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
library('CalibrationCurves')

tr1 = read.csv('M1_tr_1_12.csv')
tr2 = read.csv('M2_tr_1_12.csv')
tr3 = read.csv('M3_tr_1_12.csv')
tr4 = read.csv('M4_tr_1_12.csv')

test1 = read.csv('M1_test_1_12.csv')
test2 = read.csv('M2_test_1_12.csv')
test3 = read.csv('M3_test_1_12.csv')
test4 = read.csv('M4_test_1_12.csv')

logit = function(p){  log(p /(1-p))}
expit = function(p){  exp(p)/ (1+exp(p)) }

Pt1 = data.frame(pid = test1$patient_id, p = test1$probs, y = ifelse(test1$golds=="True", 1, 0)  )
Pt2 = data.frame(pid = test2$patient_id,p =  test2$probs, y = ifelse(test2$golds=="True", 1, 0)  )
Pt3 = data.frame(pid = test3$patient_id,p =  test3$probs, y = ifelse(test3$golds=="True", 1, 0)  )
Pt4 = data.frame(pid = test4$patient_id,p =  test4$probs, y = ifelse(test4$golds=="True", 1, 0)  )

PT1 = data.frame(pid = tr1$patient_id, p = tr1$probs, y = ifelse(tr1$golds=="True", 1, 0)  )
PT2 = data.frame(pid = tr2$patient_id,p =  tr2$probs, y = ifelse(tr2$golds=="True", 1, 0)  )
PT3 = data.frame(pid = tr3$patient_id,p =  tr3$probs, y = ifelse(tr3$golds=="True", 1, 0)  )
PT4 = data.frame(pid = tr4$patient_id,p =  tr4$probs, y = ifelse(tr4$golds=="True", 1, 0)  )

spline_pred = function(Y, lo_test ){
  # for logits
  spline = smooth.spline(x = Y$l, y = Y$y,all.knots = T, w = wts   )
  predict(spline, lo_test)
}
spl_pred = function(Y, p_test, wts=rep(1, nrow(Y)) ){
  spline = smooth.spline(x = Y$p, y = Y$y,all.knots = T, w = wts   )
  predict(spline, p_test)
}

Ytr = PT1
Ytest = Pt1

make_spline = function(Ytr, Ytest){
  
  #pos_dev = Ydev [Ydev$y ==1,] 
  #neg_dev = Ydev [Ydev$y ==0,] 
  #sub_neg_dev = neg_dev[sample(nrow(pos_dev)), ]
  #ratio_of_pos = nrow(pos_dev) / nrow(neg_dev)
  #balanced_dev = rbind( pos_dev, sub_neg_dev  )
  #weights = c(rep(1, nrow(pos_dev)),  rep(   (1/ratio_of_pos), nrow(sub_neg_dev)))
  
  pred_subset = spl_pred( Ytr,  Ytest$p) #, wts = weights)
  sub_predicted = pred_subset$y
  Pred = data.frame(p =   (sub_predicted) , y = Ytest$y)
  Pred 
}


Pred_T4_1 = make_spline(PT4, Pt4)
calibration_plot(data = Pred_T4_1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))



Pred_T1 = make_spline(PT1, Pt1)
calibration_plot(data = Pred_T1, obs = 'y', pred = 'p', x_lim = c(0,1), y_lim = c(0,1))




















cal_plots = function(){
    Cal1  = calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)")
    Cal2  = calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")
    Cal3 = calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")
    Cal4 = calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")
     
    G12 = grid.arrange(Cal1$calibration_plot, Cal2$calibration_plot, nrow =1)
    G34 = grid.arrange(Cal3$calibration_plot, Cal4$calibration_plot, nrow = 1)
    grid.arrange(G12, G34, nrow = 2)
    
    Ptot =  rbind(P1, P2, P3, P4)
    Cal.tot = calibration_plot(data = Ptot, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), 
                               xlab = "Binned Pred (All Intervals)", nTiles =20)
    
    Cal.tot
    
    
    Cal1  = calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =20)
    Cal2  = calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)", nTiles =20)
    Cal3 = calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)",nTiles =20)
    Cal4 = calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)",nTiles =20)
    
    
    G12 = grid.arrange(Cal1$calibration_plot, Cal2$calibration_plot, nrow =1)
    G34 = grid.arrange(Cal3$calibration_plot, Cal4$calibration_plot, nrow = 1)
    
    #grid.arrange(grobs = lapply(pl, "+", margin))
    
    grid.arrange(G12, G34, nrow = 2)
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
     
    #setwd('//vacrrdevmavdi01.vha.med.va.gov/MAVDEV1/Fillmore_Cancer/cat/Users/data/derived/vte_ml/vte_ml/data_for_ang/Predictions_1_14')
    #write.csv(P1, 'pred_T1.csv')
    #write.csv(P2, 'pred_T2.csv')
    #write.csv(P3, 'pred_T3.csv')
    #write.csv(P4, 'pred_T4.csv')
    
    #setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
    
    calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
    #calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
    #calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
    #calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
 } 