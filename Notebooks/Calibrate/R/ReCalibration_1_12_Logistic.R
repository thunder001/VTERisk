
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
library(glmnet)
rm(list = ls())
setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
library('CalibrationCurves')

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
Ms = split_w_rep(Mtr)
Y1 = Ms$Y1
Y2 = Ms$Y2
Y3 = Ms$Y3
Y4 = Ms$Y4 
make_P = function(Tr){
  data.frame(pid = Tr$patient_id, 
             p = Tr$probs, y = ifelse(Tr$golds=="True", 1, 0)  )
  
}
P1 = make_P(Y1)
P2 = make_P(Y2)
P3 = make_P(Y3)
P4 = make_P(Y4)
  
plot_cal = function(){
  Cal1  = calibration_plot(data = P1, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)")
  Cal2  = calibration_plot(data = P2, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")
  Cal3 = calibration_plot(data = P3, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")
  Cal4 = calibration_plot(data = P4, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")
  G12 = grid.arrange(Cal1$calibration_plot, Cal2$calibration_plot, nrow =1)
  G34 = grid.arrange(Cal3$calibration_plot, Cal4$calibration_plot, nrow = 1)
  grid.arrange(G12, G34, nrow = 2)
  #Ptot =  rbind(P1, P2, P3, P4)
  P.rep = make_P(Mtr)
  Cal.tot = calibration_plot(data = P.rep, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), 
                             xlab = "Binned Pred (All Intervals)", nTiles =20)
  Cal.tot
}
write_curves = function(){ 
      
    #setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
    setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_Python/')
    
    #library('CalibrationCurves')
    
    tr1 = write.csv(P1, 'M1_tr_1_12.csv')
    tr2 = write.csv(P2, 'M2_tr_1_12.csv')
    tr3 = write.csv(P3, 'M3_tr_1_12.csv')
    tr4 = write.csv(P4, 'M4_tr_1_12.csv')

}
 #test set
tP1 = data.frame(pid = test1$patient_id, p = test1$probs, y = ifelse(test1$golds=="True", 1, 0)  )
tP2 = data.frame(pid = test2$patient_id,p =  test2$probs, y = ifelse(test2$golds=="True", 1, 0)  )
tP3 = data.frame(pid = test3$patient_id,p =  test3$probs, y = ifelse(test3$golds=="True", 1, 0)  )
tP4 = data.frame(pid = test4$patient_id,p =  test4$probs, y = ifelse(test4$golds=="True", 1, 0)  )

frequency_analysis = function(P){
  
  Reps = data.frame( table( P$pid ) )
  Reps$pid = as.character(Reps$Var1) %>%as.integer()
  
  P. = P[!duplicated(P),]
  
  P.id = join(P., Reps)
  
  
  Pos = P.id[P.id$y ==1,]
  Neg = P.id[P.id$y ==0,]
  Neg1 = Neg[Neg$Freq>1,]
  
  par(mfrow = c(1,3))
  
  plot(Pos$Freq, (Pos$p))
  plot(Neg$Freq, (Neg$p))
  plot(Neg1$Freq, (Neg1$p))

  #plot(Neg$Freq, (Neg$p))
  
  mean(Pos$Freq)
  mean(Neg$Freq)
 
  
  binned_operations = function(Pos, func){
      c(func(Pos$Freq[Pos$p<.1]),
        func(Pos$Freq[Pos$p>.1 & Pos$p<.2]),
        func(Pos$Freq[Pos$p>.2 & Pos$p<.3]),
        func(Pos$Freq[Pos$p>.3 & Pos$p<.4]),
        func(Pos$Freq[Pos$p>.4 & Pos$p<.5]),
        func(Pos$Freq[Pos$p>.5 & Pos$p<.6]))
  }
  bin_ops = function(){
    
      binned_operations(Pos, mean)
      binned_operations(Pos, length)
      binned_operations(Pos, sd)
      binned_operations(Pos, median)
    
      binned_operations(Neg, mean)
      binned_operations(Neg, length)
      binned_operations(Neg, sd)
      binned_operations(Neg, median)
    
      binned_operations(Neg1, mean)
      binned_operations(Neg1, length)
      binned_operations(Neg1, sd)
      binned_operations(Neg1, median)
    }
  
  mean(Neg$Freq==1)
  mean(Pos$Freq==1)
  
  summary( lm( Pos$p ~ Pos$Freq))
  summary( lm( Neg$p ~ Neg$Freq))
  
  plot(Neg$Freq, Neg$p)
  
  plot(( Pos$Freq), (Pos$p))
  hist(( Pos$Freq) )

  mean(( Pos$Freq ) )
  sd(  ( Pos$Freq))
  
  mean( log( Pos$Freq))
   
}

Mdev = read.csv('totM_dev_1_12.csv')
Mdev$X = NULL

Mdevs = split_w_rep(Mdev)
Pdev1 = make_P(Mdevs$Y1)
Pdev2 = make_P(Mdevs$Y2)
Pdev3 = make_P(Mdevs$Y3)
Pdev4 = make_P(Mdevs$Y4)
#Neg_Lambda = optimize(getpoi, Neg$Freq)
getpoi = function(lambda) abs(ppois(2,  lambda) - .596)

redo_frequency_on_dev = function(Pdev,P,  d1 = 1, d2 = 1, eps = -1, inflate_neg=F ){

  Reps = data.frame( table( P$pid ) )
  Reps$pid = as.character(Reps$Var1) %>%as.integer()
  P. = P[!duplicated(P),]
  P.id = join(P., Reps)
  Pos = P.id[P.id$y ==1,]
  Neg = P.id[P.id$y ==0,]
  
  pos_freq_dist = rnorm(n = nrow(Pdev[Pdev$y==1,]),  mean(Pos$Freq) + d1, sd(Pos$Freq) + d2)
  pos_freq_dist [pos_freq_dist<1] = 1
  Neg_Lambda = optimize(getpoi, Neg$Freq) 
  
 
  pois_freq_dist = rpois(n = nrow(Pdev[Pdev$y==0,]), lambda = Neg_Lambda$minimum + eps) 
    
  if(inflate_neg==TRUE){
    neg_freq_dist =  pois_freq_dist
  }else{
    neg_freq_dist = ifelse(  rbinom(nrow(Pdev[Pdev$y==0,]), 
                                    1,  
                                    mean(Neg$Freq==1)), 1, pois_freq_dist+1)    
    
  }
  Pdev$Freq[Pdev$y==1]  = round(pos_freq_dist) 
  Pdev$Freq [Pdev$y == 0]= neg_freq_dist
  dev_freqs = Pdev$Freq
  Repdev = Pdev %>% slice(rep(1:n(), times = dev_freqs))
  
  return(Repdev)
}


try_I1 = function(){
  
  
  cv_logit_1 = function(Ptrain, test ){
    Ptrain$pos_freq = ifelse(Ptrain$y==1,Ptrain$Freq, 1 )
    train_cv = cv.glmnet(x = cbind(1, logit(Ptrain$p) ),family = 'binomial', y = Ptrain$y,  weights = (1/Ptrain$Freq)^1.25,
                         type.measure = 'mse' , intercept = FALSE , nfolds = 20 )
    cv_pred = predict(train_cv, newx = cbind(1,  logit (test$probs)), type = 'response' )
    cv_pred
  }
  
  
  Dev1_bal = redo_frequency_on_dev(Pdev1, P1, d1 = 3, d2= - .5, eps = -1, inflate_neg = TRUE)
  cv1_test = cv_logit (Dev1_bal, test1) 
  P.cv =  data.frame(p = as.numeric(cv1_test), y = ifelse(test1$golds=="True", 1, 0)  )
  calibration_plot(data = P.cv, obs = 'y', pred = 'p', x_lim = c(0,.25), y_lim = c(0,.25), 
                   xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =10)
  

  
  }



Dev1_balanced = redo_frequency_on_dev(Pdev1, P1, eps = .5)
Dev2_balanced = redo_frequency_on_dev(Pdev2, P2)
Dev3_balanced = redo_frequency_on_dev(Pdev3, P3)
Dev4_balanced = redo_frequency_on_dev(Pdev4, P4)


cv_logit = function(Ptrain, test ){
  Ptrain$pos_freq = ifelse(Ptrain$y==1,Ptrain$Freq, 1 )
  train_cv = cv.glmnet(x = cbind(1, logit(Ptrain$p) ),family = 'binomial', y = Ptrain$y,  weights = (1/Ptrain$Freq),
                       type.measure = 'mae' , intercept = FALSE , nfolds = 20 )
  cv_pred = predict(train_cv, newx = cbind(1,  logit (test$probs)), type = 'response' )
  cv_pred
}

cv1 = cv_logit (Dev1_balanced, test1) 
P.cv1 =  data.frame(p = as.numeric(cv1), y = ifelse(test1$golds=="True", 1, 0)  )
tCal1  = calibration_plot(data = P.cv1, obs = 'y', pred = 'p', x_lim = c(0,.2), y_lim = c(0,.2), 
                          xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =10)

cv2 = cv_logit (Dev2_balanced, test2) 
P.cv2 =  data.frame(p = as.numeric(cv2), y = ifelse(test2$golds=="True", 1, 0)  )
tCal2  = calibration_plot(data = P.cv2, obs = 'y', pred = 'p', x_lim = c(0,.1), y_lim = c(0,.1), 
                          xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")

cv3 = cv_logit (Dev3_balanced, test3) 
P.cv3 =  data.frame(p = as.numeric(cv3), y = ifelse(test3$golds=="True", 1, 0)  )
tCal3  = calibration_plot(data = P.cv3, obs = 'y', pred = 'p', x_lim = c(0,.08), y_lim = c(0,.08),
                          xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")

cv4 = cv_logit (Dev4_balanced, test4) 
P.cv4 =  data.frame(p = as.numeric(cv4), y = ifelse(test4$golds=="True", 1, 0)  )
tCal4  = calibration_plot(data = P.cv4, obs = 'y', pred = 'p', x_lim = c(0,.08), y_lim = c(0,.08), 
                          xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")

G12 = grid.arrange(tCal1$calibration_plot, tCal2$calibration_plot, nrow =1)
G34 = grid.arrange(tCal3$calibration_plot, tCal4$calibration_plot, nrow = 1)
grid.arrange(G12, G34, nrow = 2)


















Ptot =  rbind(P1, P2, P3, P4)
Cal.tot = calibration_plot(data = Ptot, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), 
                           xlab = "Binned Pred (All Intervals)", nTiles =20)

Cal.tot



plot_verify = function(){
  
  
  par(mfrow = c(2,2))
  plot(Neg$Freq, (Neg$p))
  plot(Pos$Freq, (Pos$p))
  
  plot(Pdev$Freq[Pdev$y==0],  Pdev$p[Pdev$y==0])
  plot(Pdev$Freq[Pdev$y==1],  Pdev$p[Pdev$y==1])
  
  
  
  
  hist(Pdev$Freq[Pdev$y==1])
  hist(Pdev$Freq[Pdev$y==0])
  
}


check_1 = function(){
  
  par(mfrow = c(1,2))
  hist( dev_freqs)
  hist(Reps$Freq)
  
  quantile( dev_freqs)
  quantile(Reps$Freq)
  
}



 








library(doBy)
 





######## test set

test1 = read.csv('M1_test_1_12.csv')
test2 = read.csv('M2_test_1_12.csv')
test3 = read.csv('M3_test_1_12.csv')
test4 = read.csv('M4_test_1_12.csv')

logit = function(p){  log(p /(1-p))}
expit = function(p){  exp(p)/ (1+exp(p)) }


tP1 = data.frame(pid = test1$patient_id, p = test1$probs, y = ifelse(test1$golds=="True", 1, 0)  )
tP2 = data.frame(pid = test2$patient_id,p =  test2$probs, y = ifelse(test2$golds=="True", 1, 0)  )
tP3 = data.frame(pid = test3$patient_id,p =  test3$probs, y = ifelse(test3$golds=="True", 1, 0)  )
tP4 = data.frame(pid = test4$patient_id,p =  test4$probs, y = ifelse(test4$golds=="True", 1, 0)  )

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

grid.arrange(G12, G34, nrow = 2)


library(glmnet)

train1_cv = cv.glmnet(x = cbind(1  , logit(P1$p) ),family = 'binomial', y = P1$y, 
                      type.measure = 'mae')
cv_pred1 = predict(train1_cv, newx = cbind(1, logit(test1$probs)), type = 'response')


cv_logit = function(Ptrain, test ){
  train_cv = cv.glmnet(x = cbind(1  , logit(Ptrain$p) ),family = 'binomial', y = Ptrain$y, 
                        type.measure = 'deviance')
  cv_pred = predict(train_cv, newx = cbind(1, logit(test$probs)), type = 'response')
  cv_pred
}

cv1 = cv_logit (P1, test1)
hist(cv1)
cv2 = cv_logit (P2, test2)
hist(cv2)
cv3 = cv_logit (P3, test3)
hist(cv3)
cv4 = cv_logit (P4, test4)

make_P = function(Tr){
  data.frame(pid = Tr$patient_id, 
             p = Tr$probs, y = ifelse(Tr$golds=="True", 1, 0)  )
  
}

P.cv1 =  data.frame(p = as.numeric(cv1), y = ifelse(test1$golds=="True", 1, 0)  )
P.cv2 =  data.frame(p = as.numeric(cv2), y = ifelse(test2$golds=="True", 1, 0)  )

Cv.Cal1  = calibration_plot(data = P.cv1, obs = 'y', pred = 'p', x_lim = c(0,.7), y_lim = c(0,.7), 
                            xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =20)

Cv.Cal2  = calibration_plot(data = P.cv2, obs = 'y', pred = 'p', x_lim = c(0,.7), y_lim = c(0,.7), 
                            xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =20)
max(cv1)
Cv.Cal1
#Cv.Cal2







expit( cv_pred1)

confusion_part = function(){

    
    
    confu = function(p, yy, THRESH=.5){
      CM = confusionMatrix( ifelse(p > THRESH,1,0) %>% as.factor(), yy %>% as.factor())
      t(CM$table)
    }
    
    train3_cv = cv.glmnet(x = cbind(1,devs$Y3$l ) , y =devs$Y3$y , weights = (ww3$w3))
    
    
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
