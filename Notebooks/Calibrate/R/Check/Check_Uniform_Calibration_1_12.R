
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


Mtr = read.csv('totM_tr_1_12.csv')

Mtr$X = NULL

Mtr0 = Mtr[!duplicated(Mtr),]

splitt = function(MD){
  
  MD = MD[order(MD$patient_id),] 
  MD$X = MD$patient_id
  MD$pids = NULL
  #MD$patient_id = NULL
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
  
  return(list(#Y1=select(Y1, X, p,    y ), Y2=select(Y2, X, p,    y ),  Y3=select(Y3, X, p,    y ), Y4=select(Y4, X, p,    y ), 
              Y1 = Y1, Y2 = Y2, Y3= Y3, Y4=Y4))
  
}

M0s = splitt(Mtr0)
mean(M0s$Y1$golds=="True" )
mean(M0s$Y2$golds=="True" )
mean(M0s$Y3$golds=="True" )
mean(M0s$Y4$golds=="True" )

MD = Mtr

library('scrutiny')

split_w_rep = function(MD){
  
  MD = MD[order(MD$patient_id),] 
  MD$pids = NULL
  MD$dob = NULL
  MD$days_from_outcome = (as.Date(MD$outcome_date) - as.Date(MD$date)) %>% as.numeric()
  #MD$diag_date = NULL
  MD$daysfrom_index = (as.Date(MD$date) - as.Date(MD$index_date)) %>% as.numeric()
  MD$y =  ifelse(MD$golds == "True", 1, 0)
  MD$p = MD$probs #/ max(MD$probs)
  MD$censor_times = NULL
  
   #MD$Diag_to_Index =(MD$index_date%>%as.Date() - MD$diag_date%>%as.Date())%>% as.numeric()
  #X = select(MD,  pid =X,  "outcome_date" ,   "index_date",   "diag_date"  ,"outcome" ,  "y"  ,predicted_prob="probs"    ,  
  #           model_date ="dates",  "days_from_outcome",  "daysfrom_index"   )

  D = MD[order(  MD$patient_id, MD$dates   ),] 
  
  D$patient_golds = NULL
  
  D$days_to_final_censors = NULL
  
  D$Pat_Date_dup = duplicated(cbind(D$patient_id, D$dates))

  D_by_Pat = split(D, D$patient_id)
  
  
  S = D_by_Pat[['1000654643']]
  
  getI = function(S){
    S$I = cumsum( !S$Pat_Date_dup )
    S
  }
  
  D_I_by_Pat = lapply(D_by_Pat, function(S)   getI(S))
  
  D_re = do.call(rbind, D_I_by_Pat)
  
  Y1 = D_re [D_re$I ==1,]
  Y2 = D_re [D_re$I ==2,]
  Y3 = D_re [D_re$I ==3,]
  Y4 = D_re [D_re$I ==4,]
  
  return(list(#Y1=select(Y1, X, p,    y ), Y2=select(Y2, X, p,    y ),  Y3=select(Y3, X, p,    y ), Y4=select(Y4, X, p,    y ), 
    Y1 = Y1, Y2 = Y2, Y3= Y3, Y4=Y4))
  
}


Ms = split(Mtr0)


Y1 = Ms$Y1
Y2 = Ms$Y2
Y3 = Ms$Y3
Y4 = Ms$Y4

sum(Y1$y)
sum(Y2$y)
sum(Y3$y)
sum(Y4$y)



mean(Y1$y)
mean(Y2$y)
mean(Y3$y)
mean(Y4$y)


make_P = function(Tr){
  data.frame(pid = Tr$patient_id, 
             p = Tr$probs, y = ifelse(Tr$golds=="True", 1, 0)  )
  
}


troubleshoot_tr = function(tr){
 
  
  P1 = make_P(Y1)
  P2 = make_P(Y2)
  P3 = make_P(Y3)
  P4 = make_P(Y4)
  
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
  
  P.rep = make_P(Mtr)
  Cal.tot = calibration_plot(data = P.rep, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), 
                             xlab = "Binned Pred (All Intervals)", nTiles =20)
  Cal.tot
  
  
}

#setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_Python/')

#library('CalibrationCurves')

tr1 = write.csv(P1, 'M1_tr_1_12.csv')
tr2 = write.csv(P2, 'M2_tr_1_12.csv')
tr3 = write.csv(P3, 'M3_tr_1_12.csv')
tr4 = write.csv(P4, 'M4_tr_1_12.csv')










test1 = read.csv('M1_test_1_12.csv')
test2 = read.csv('M2_test_1_12.csv')
test3 = read.csv('M3_test_1_12.csv')
test4 = read.csv('M4_test_1_12.csv')

logit = function(p){  log(p /(1-p))}
expit = function(p){  exp(p)/ (1+exp(p)) }

par(mfrow = c(1,4))

if(train){
  P1 = data.frame(pid = tr1$patient_id, p = tr1$probs, y = ifelse(tr1$golds=="True", 1, 0)  )
  P2 = data.frame(pid = tr2$patient_id,p =  tr2$probs, y = ifelse(tr2$golds=="True", 1, 0)  )
  P3 = data.frame(pid = tr3$patient_id,p =  tr3$probs, y = ifelse(tr3$golds=="True", 1, 0)  )
  P4 = data.frame(pid = tr4$patient_id,p =  tr4$probs, y = ifelse(tr4$golds=="True", 1, 0)  )
}else{
  P1 = data.frame(pid = test1$patient_id, p = test1$probs, y = ifelse(test1$golds=="True", 1, 0)  )
  P2 = data.frame(pid = test2$patient_id,p =  test2$probs, y = ifelse(test2$golds=="True", 1, 0)  )
  P3 = data.frame(pid = test3$patient_id,p =  test3$probs, y = ifelse(test3$golds=="True", 1, 0)  )
  P4 = data.frame(pid = test4$patient_id,p =  test4$probs, y = ifelse(test4$golds=="True", 1, 0)  )
}




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




#setwd('//vacrrdevmavdi01.vha.med.va.gov/MAVDEV1/Fillmore_Cancer/cat/Users/data/derived/vte_ml/vte_ml/data_for_ang/Predictions_1_14')
#write.csv(P1, 'pred_T1.csv')
#write.csv(P2, 'pred_T2.csv')
#write.csv(P3, 'pred_T3.csv')
#write.csv(P4, 'pred_T4.csv')

#setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')

#calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
#calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
#calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
#calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.5), y_lim = c(0,.5))
 
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
