
message(normalizePath("~"))
rm(list = ls())

logit = function(p){  log(p /(1-p))}
expit = function(p){  exp(p)/ (1+exp(p)) }
library(fst)
library(dplyr)
library(stringr)
library(plyr)
library(data.table)
library(caret)
library("Metrics")
library(egg)
library(glmnet)
library(predtools)
library('CalibrationCurves')

setwd('G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/')
Mtr = read.csv('totbM_tr_1_28.csv')
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

Mtest = read.csv('Mb_test_1_28.csv')
Mtest$X = NULL

Mtests = split_w_rep(Mtest)

R_tP01 = make_P(Mtests$Y1)
R_tP02 = make_P(Mtests$Y2)
R_tP03 = make_P(Mtests$Y3)
R_tP04 = make_P(Mtests$Y4)


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

Mdev = read.csv('totbM_dev_1_28.csv')
Mdev$X = NULL
Mdevs = split_w_rep(Mdev)
Pdev1 = make_P(Mdevs$Y1)
Pdev2 = make_P(Mdevs$Y2)
Pdev3 = make_P(Mdevs$Y3)
Pdev4 = make_P(Mdevs$Y4)
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
first_plot_Dev_balanced = function(Dev1_balanced, Dev2_balanced, Dev3_balanced, Dev4_balanced){ 
  
  balP1 = data.frame(  p = Dev1_balanced$p, y = Dev1_balanced$y  )
  balP2 = data.frame(p =  Dev2_balanced$p, y = Dev2_balanced$y )
  balP3 = data.frame(p =  Dev3_balanced$p, y = Dev3_balanced$y  )
  balP4 = data.frame(p =  Dev4_balanced$p, y =Dev4_balanced$y  )
  
  dCal1  = calibration_plot(data = balP1, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)")
  dCal2  = calibration_plot(data = balP2, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")
  dCal3 = calibration_plot(data = balP3, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")
  dCal4 = calibration_plot(data = balP4, obs = 'y', pred = 'p', x_lim = c(0,.4), y_lim = c(0,.4), xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")
  
  G12 = grid.arrange(dCal1$calibration_plot, dCal2$calibration_plot, nrow =1)
  G34 = grid.arrange(dCal3$calibration_plot, dCal4$calibration_plot, nrow = 1)
  grid.arrange(G12, G34, nrow = 2)
}

Dev1_balanced0 = redo_frequency_on_dev(Pdev1, P1, d1 = 2, d2 = .5, eps = -.5)
Dev2_balanced0 = redo_frequency_on_dev(Pdev2, P2,d1 = 2, d2 = .5, eps = -.5)
Dev3_balanced0 = redo_frequency_on_dev(Pdev3, P3,d1 = 2, d2 = .5, eps = -.5)
Dev4_balanced0 = redo_frequency_on_dev(Pdev4, P4,d1 = 5, d2 = -.5, eps = -1)
first_plot_Dev_balanced (Dev1_balanced0, Dev2_balanced0, Dev3_balanced0, Dev4_balanced0)


Dev1_balanced = redo_frequency_on_dev(Pdev1, P1, d1 = 2.5, d2 = .25, eps = -.5)
Dev2_balanced = redo_frequency_on_dev(Pdev2, P2,d1 = 3.25, d2 = -.5, eps = -.5)
Dev3_balanced = redo_frequency_on_dev(Pdev3, P3,d1 = 2, d2 = .25, eps = -.5)
Dev4_balanced = redo_frequency_on_dev(Pdev4, P4,d1 = 4, d2 = -.5, eps = -1)
first_plot_Dev_balanced (Dev1_balanced0, Dev2_balanced0, Dev3_balanced0, Dev4_balanced0)

# test others 
Dev1_balanced2 = redo_frequency_on_dev(Pdev1, P1, d1 = 5, d2 = .5, eps = -2)
Dev2_balanced2 = redo_frequency_on_dev(Pdev2, P2,d1 = 6, d2 = 1, eps = -1.3)
Dev3_balanced2 = redo_frequency_on_dev(Pdev3, P3,d1 = 5, d2 = .25, eps = -1.5) #redo_frequency_on_dev(Pdev3, P3,d1 = 4, d2 = -1, eps = -2)
Dev4_balanced2 = redo_frequency_on_dev(Pdev4, P4,d1 = 5, d2 = -.5, eps = -1) 
#redo_frequency_on_dev(Pdev4, P4,d1 = 3, d2 = 2, eps = -2.3)

#Dev4_balanced2
first_plot_Dev_balanced (Dev1_balanced2, Dev2_balanced2,    Dev3_balanced2, Dev4_balanced2)

spline_pred = function(Ptrain, test , knots = 15, pen = .75 ){
  Ptrain$l = logit(Ptrain$p)
  test$l = logit(test$p)
  spline = smooth.spline(x = Ptrain$l, y = Ptrain$y, w = 1/Ptrain$Freq, nknots=knots, penalty = pen  )
  spl_pred = predict(spline, test$l)
  spl_pred$y
}

cv1 = spline_pred (Dev1_balanced2, R_tP01) 
P.cv1 =  data.frame(p = as.numeric(cv1), y =  R_tP01$y  )
tCal1  = calibration_plot(data = P.cv1, obs = 'y', pred = 'p', x_lim = c(0,.1), y_lim = c(0,.1), 
                          xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =10)


cv2 = spline_pred (Dev2_balanced2, R_tP02, knots = 14, pen =  .1) 
P.cv2 =  data.frame(p = as.numeric(cv2), y =  R_tP02$y  )
tCal2  = calibration_plot(data = P.cv2, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                          xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")


cv3 = spline_pred (Dev3_balanced2, R_tP03) 
P.cv3 =  data.frame(p = as.numeric(cv3), y  =  R_tP03$y  )
tCal3  = calibration_plot(data = P.cv3, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05),
                          xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")

cv4 = spline_pred (Dev4_balanced2, R_tP04, knots = 5, pen = .01) 
P.cv4 =  data.frame(p = as.numeric(cv4), y  =  R_tP04$y  )
tCal4  = calibration_plot(data = P.cv4, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                          xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")




G12 = grid.arrange(tCal1$calibration_plot, tCal2$calibration_plot, nrow =1)
G34 = grid.arrange(tCal3$calibration_plot, tCal4$calibration_plot, nrow = 1)
grid.arrange(G12, G34, nrow = 2)
 

c(pROC::auc (P.cv1$y, P.cv1$p ), pROC::auc (P.cv2$y,  P.cv2$p), 
  pROC::auc (P.cv3$y,  P.cv3$p), pROC::auc (P.cv4$y,  P.cv4$p))

c(pROC::auc (P.cv1$y,  R_tP01$p),
  pROC::auc (P.cv2$y,  R_tP02$p),
  pROC::auc (P.cv3$y,  R_tP03$p), 
  pROC::auc (P.cv4$y,  R_tP04$p )  )


 
testing_if_match = function(){
  test_tot = rbind(test1, test2, test3, test4)
  sum( test_tot$golds=="True")
  pROC::auc (test_tot$golds=="True", test_tot$probs )

  
  pROC::auc (test1$golds=="True", test1$probs )
  pROC::auc (test2$golds=="True", test2$probs )
  pROC::auc (test3$golds=="True", test3$probs )
  pROC::auc (test4$golds=="True", test4$probs )
  
    
}
spline_pred_dev = function(){
  
  cv1 = spline_pred (Dev1_balanced, Mdevs$Y1) 
  P.cv1 =  data.frame(p = as.numeric(cv1), y = ifelse(Mdevs$Y1$golds=="True", 1, 0)  )
  dCal1  = calibration_plot(data = P.cv1, obs = 'y', pred = 'p', x_lim = c(0,.1), y_lim = c(0,.1), 
                            xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =10)
  
  cv2 = spline_pred (Dev2_balanced,  Mdevs$Y2,  knots = 12, pen = 1.95) 
  P.cv2 =  data.frame(p = as.numeric(cv2), y = ifelse( Mdevs$Y2$golds=="True", 1, 0)  )
  tCal2  = calibration_plot(data = P.cv2, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                            xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")
  tCal2
  
  cv3 = spline_pred (Dev3_balanced,  Mdevs$Y3) 
  P.cv3 =  data.frame(p = as.numeric(cv3), y = ifelse( Mdevs$Y3$golds=="True", 1, 0)  )
  tCal3  = calibration_plot(data = P.cv3, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05),
                            xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")
  
  cv4 = spline_pred (Dev4_balanced, test4, knots = 29, pen = .01) 
  P.cv4 =  data.frame(p = as.numeric(cv4), y = ifelse(test4$golds=="True", 1, 0)  )
  tCal4  = calibration_plot(data = P.cv4, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                            xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")
  #tCal4
  
}

try_I1_Temperation = function(){
  
  
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

confu = function(p, yy, THRESH=.5){
  CM = confusionMatrix( ifelse(p > THRESH,1,0) %>% as.factor(), yy %>% as.factor())
  t(CM$table)
}

confu (P.cv1$p,  P.cv1$y,    mean ( Pdev1$y )  )
confu (P.cv2$p,  P.cv2$y,    mean ( Pdev2$y )  )
confu (P.cv3$p,  P.cv3$y,    mean ( Pdev3$y )  )
confu (P.cv4$p,  P.cv4$y,    mean ( Pdev4$y )  )

confu (P.cv1$p,  P.cv1$y,    mean ( P.cv1$y )  )
confu (P.cv2$p,  P.cv2$y,    mean ( P.cv2$y )  )
confu (P.cv3$p,  P.cv3$y,    mean ( P.cv3$y )  )
confu (P.cv4$p,  P.cv4$y,    mean ( P.cv4$y )  )


confu (P.cv1$p,  P.cv1$y,    mean (  P1[!duplicated(P1),]$y )  )
confu (P.cv2$p,  P.cv2$y,    mean (  P2[!duplicated(P2),]$y )  )
confu (P.cv3$p,  P.cv3$y,    mean (  P3[!duplicated(P3),]$y )  )
confu (P.cv4$p,  P.cv4$y,    mean (  P4[!duplicated(P4),]$y )  )



extra_functions(){
  
  
  confu (P.cv1$p,  P.cv1$y,    mean (  P1[!duplicated(P1),]$y )  )
  confu (P.cv2$p,  P.cv2$y,    mean (  P2[!duplicated(P2),]$y )  )
  confu (P.cv3$p,  P.cv3$y,    mean (  P3[!duplicated(P3),]$y )  )
  confu (P.cv4$p,  P.cv4$y,    mean (  P4[!duplicated(P4),]$y )  )
  spline_pred_dev = function(){
    
    cv1 = spline_pred (Dev1_balanced, Mdevs$Y1) 
    P.cv1 =  data.frame(p = as.numeric(cv1), y = ifelse(Mdevs$Y1$golds=="True", 1, 0)  )
    dCal1  = calibration_plot(data = P.cv1, obs = 'y', pred = 'p', x_lim = c(0,.1), y_lim = c(0,.1), 
                              xlab = "Binned Pred (Interval 1: 0-3 Months after Index Date)", nTiles =10)
    
    cv2 = spline_pred (Dev2_balanced,  Mdevs$Y2,  knots = 12, pen = 1.95) 
    P.cv2 =  data.frame(p = as.numeric(cv2), y = ifelse( Mdevs$Y2$golds=="True", 1, 0)  )
    tCal2  = calibration_plot(data = P.cv2, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                              xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")
    tCal2
    
    cv3 = spline_pred (Dev3_balanced,  Mdevs$Y3) 
    P.cv3 =  data.frame(p = as.numeric(cv3), y = ifelse( Mdevs$Y3$golds=="True", 1, 0)  )
    tCal3  = calibration_plot(data = P.cv3, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05),
                              xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")
    
    cv4 = spline_pred (Dev4_balanced, test4, knots = 29, pen = .01) 
    P.cv4 =  data.frame(p = as.numeric(cv4), y = ifelse(test4$golds=="True", 1, 0)  )
    tCal4  = calibration_plot(data = P.cv4, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                              xlab = "Binned Pred (Interval 4: 9-12 Months after Index Date)")
    #tCal4
    
  }
  
  try_I1_Temperation = function(){
    
    
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
  
  
  different_confusions = function(){
    
    
    # M train raw, no balancing
    Mtr_nobal = read.csv('nobalM_tr_1_12.csv')
    Mtr_nobal$X = NULL
    M00s = splitt(Mtr_nobal)
    
    mean(M00s$Y1$golds=="True" )
    mean(M00s$Y2$golds=="True" )
    mean(M00s$Y3$golds=="True" )
    mean(M00s$Y4$golds=="True" )
    
    rate1 =tr.v1/ 48800
    rate2=tr.v2/42000
    rate3=tr.v3/38000
    rate4=tr.v4/35000
    
    rate1 = mean ( Pdev1$y ) 
    rate2= mean ( Pdev2$y ) 
    rate3= mean ( Pdev3$y ) 
    rate4= mean ( Pdev4$y ) 
    
    
    # good: just use mean between these, no sqrt
    rate1 =  (   mean(M00s$Y1$golds=="True" )^1   ) 
    rate2 =  (   mean(M00s$Y2$golds=="True" )^1   ) 
    rate3 =  (   mean(M00s$Y3$golds=="True" )^1   ) 
    rate4 =  (    mean(M00s$Y4$golds=="True" )^1 ) 
    
    
    harmoni = function(x,y) 2*x*y/(x+y)
    rate1 =  harmoni(  mean(M00s$Y1$golds=="True" )  ,  mean ( P.cv1$y) ) 
    rate2 =  harmoni(  mean(M00s$Y2$golds=="True" )  ,  mean ( P.cv2$y) ) 
    rate3 =  harmoni(  mean(M00s$Y3$golds=="True" )  ,  mean ( P.cv3$y)) 
    rate4 =  harmoni(  mean(M00s$Y4$golds=="True" )  ,  mean ( P.cv4$y) ) 
    
    rate1 =  ((mean(M00s$Y1$golds=="True" )^2 +  mean ( P.cv1$y)^2  )/2 )^.5
    rate2 =  ((mean(M00s$Y2$golds=="True" )^2 +  mean ( P.cv2$y )^2  )/2 )^.5
    rate3 =  ((mean(M00s$Y3$golds=="True" )^2 +  mean ( P.cv3$y )^2  )/2 )^.5
    rate4 =  ((mean(M00s$Y4$golds=="True" )^2 +  mean ( P.cv4$y )^2  )/2 )^.5
    
    
    
    confu (P.cv1$p,  P.cv1$y,   rate1  )
    confu (P.cv2$p,  P.cv2$y,   rate2)
    confu (P.cv3$p,  P.cv3$y,   rate3)
    confu (P.cv4$p,  P.cv4$y,   rate4 )
    
    
    
    
    rate1 =  (mean (  P1[!duplicated(P1),]$y ) +  mean ( P.cv1$y ) + mean ( Pdev1$y ))/3 
    rate2 =  (mean (  P2[!duplicated(P2),]$y ) +  mean ( P.cv2$y ) + mean ( Pdev2$y ))/3 
    rate3 =  (mean (  P3[!duplicated(P3),]$y ) +  mean ( P.cv3$y ) + mean ( Pdev3$y ))/3 
    rate4 =  (mean (  P4[!duplicated(P4),]$y ) +  mean ( P.cv4$y ) + mean ( Pdev4$y ))/3 
    
    rate1 =  ((mean (  P1[!duplicated(P1),]$y )^2 +  mean ( P.cv1$y)^2 + mean ( Pdev1$y )^2)/3 )^.5
    rate2 =  ((mean (  P2[!duplicated(P2),]$y )^2 +  mean ( P.cv2$y )^2 + mean ( Pdev2$y )^2)/3 )^.5
    rate3 =  ((mean (  P3[!duplicated(P3),]$y )^2 +  mean ( P.cv3$y )^2 + mean ( Pdev3$y )^2)/3 )^.5
    rate4 =  ((mean (  P4[!duplicated(P4),]$y )^2 +  mean ( P.cv4$y )^2 + mean ( Pdev4$y )^2)/3 )^.5
    
    
    # mean of uncorrected vs balanced sample rates 
    rate1 =  ((mean (  P1[!duplicated(P1),]$y )^2 +   mean(M00s$Y1$golds=="True" )^2  )/2 )^.5
    rate2 =  ((mean (  P2[!duplicated(P2),]$y )^2 +   mean(M00s$Y2$golds=="True" )^2  )/2 )^.5
    rate3 =  ((mean (  P3[!duplicated(P3),]$y )^2 +   mean(M00s$Y3$golds=="True" )^2  )/2 )^.5
    rate4 =  ((mean (  P4[!duplicated(P4),]$y )^2 +   mean(M00s$Y4$golds=="True" )^2  )/2 )^.5
    
    # good: just use mean between these, no sqrt
    rate1 =  ((mean (  P1[!duplicated(P1),]$y )^1 +   mean(M00s$Y1$golds=="True" )^1  )/2 ) 
    rate2 =  ((mean (  P2[!duplicated(P2),]$y )^1 +   mean(M00s$Y2$golds=="True" )^1  )/2 ) 
    rate3 =  ((mean (  P3[!duplicated(P3),]$y )^1 +   mean(M00s$Y3$golds=="True" )^1  )/2 ) 
    rate4 =  ((mean (  P4[!duplicated(P4),]$y )^1 +   mean(M00s$Y4$golds=="True" )^1  )/2 ) 
    
    rate1 =  ((mean (  P1[!duplicated(P1),]$y )^1 +   mean(M00s$Y1$golds=="True" )^1  )/2 ) 
    rate2 =  ((mean (  P2[!duplicated(P2),]$y )^1 +   mean(M00s$Y2$golds=="True" )^1  )/2 ) 
    rate3 =  ((mean (  P3[!duplicated(P3),]$y )^1 +   mean(M00s$Y3$golds=="True" )^1  )/2 ) 
    rate4 =  ((mean (  P4[!duplicated(P4),]$y )^1 +   mean(M00s$Y4$golds=="True" )^1  )/2 ) 
    
    
    
    
    confu (P.cv1$p,  P.cv1$y,   .059   )
    confu (P.cv2$p,  P.cv2$y,   .059/2  )
    confu (P.cv3$p,  P.cv3$y,   .059/3)
    confu (P.cv4$p,  P.cv4$y,   .059/4  )
    
    pROC::auc (P.cv1$y,  test1$probs       )
    pROC::auc (P.cv2$y,  test2$probs      )
    pROC::auc (P.cv3$y,  test3$probs     )
    pROC::auc (P.cv4$y,  test4$probs   )
    
    
    detail_confu = function(p, yy, THRESH=.5){
      CM = confusionMatrix( ifelse(p > THRESH,1,0) %>% as.factor(), yy %>% as.factor())
      CM
    }
    
    detail_confu (P.cv1$p,  P.cv1$y,   rate1  )
    detail_confu (P.cv2$p,  P.cv2$y,   rate2)
    detail_confu (P.cv3$p,  P.cv3$y,   rate3)
    detail_confu (P.cv4$p,  P.cv4$y,   rate4 )
    
    
  }
  
  
  harmonic_mean_thresh = function(){
    
    harmoni = function(x,y) 2*x*y/(x+y)
    rate1 =  harmoni(  mean(M00s$Y1$golds=="True" )  ,  .06 ) 
    rate2 =  harmoni(  mean(M00s$Y2$golds=="True" )  ,  .03 ) 
    rate3 =  harmoni(  mean(M00s$Y3$golds=="True" )  ,  .03 /2) 
    rate4 =  harmoni(  mean(M00s$Y4$golds=="True" )  ,  .03/4 ) 
    
    rate1 =  harmoni(  mean(M00s$Y1$golds=="True" )  ,  mean (  P1[!duplicated(P1),]$y ) ) 
    rate2 =  harmoni(  mean(M00s$Y2$golds=="True" )  ,  mean (  P2[!duplicated(P2),]$y ) ) 
    rate3 =  harmoni(  mean(M00s$Y3$golds=="True" )  ,  mean (  P3[!duplicated(P3),]$y )) 
    rate4 =  harmoni(  mean(M00s$Y4$golds=="True" )  , mean (  P4[!duplicated(P4),]$y)  ) 
    
    confu (P.cv1$p,  P.cv1$y,   rate1  )
    confu (P.cv2$p,  P.cv2$y,   rate2)
    confu (P.cv3$p,  P.cv3$y,   rate3)
    confu (P.cv4$p,  P.cv4$y,   rate4 )
    
    confu (P.cv1$p,  P.cv1$y,   .06  )
    confu (P.cv2$p,  P.cv2$y,   .06/2)
    confu (P.cv3$p,  P.cv3$y,   .06/4)
    confu (P.cv4$p,  P.cv4$y,   .06/8 )
    
    
  }
  
  find_optim_threshold = function(){
    
    library(Metrics) 
    A1 = which(P.cv1$y==1)
    A2 = which(P.cv2$y==1)
    A3 = which(P.cv3$y==1)
    A4 = which(P.cv4$y==1)
    
    range =  .025+ 1:4000/40000
    
    P.cv = P.cv1
    
    make_F1_confu = function(range , P.cv, t){
      
      A = which(P.cv$y==1)
      #I_f1  = sapply( range ,  function(i)   f1( A, which( P.cv$p > i)))
      I_f2  = sapply( range ,  function(i)    fbeta_score( P.cv$y, ( P.cv$p > i), t ))
      harm =   (I_f2)
      optim_1 =range [ which.max(harm)]
      list( confu (P.cv$p,  P.cv$y,   optim_1  ), optim_1)
    }
    
    Conf1 = make_F1_confu (.025+ 1:4000/40000, P.cv1, 1)
    Conf2 = make_F1_confu (.025+ 1:4000/40000 , P.cv2,  2) 
    Conf3 = make_F1_confu (.025+ 1:4000/40000 - .025, P.cv3,3)
    Conf4 = make_F1_confu (.025+ 1:4000/40000 - .025, P.cv4, 4)
    
    Conf1
    Conf2
    Conf3
    Conf4
    
    
    
    range =  .1+ 1:2000/10000
    max(range)
    
    tConf1 = make_F1_confu (range+.1, tP1, 1)
    tConf2 = make_F1_confu (range , tP2,  2) 
    tConf3 = make_F1_confu (range - .1, tP3,3)
    tConf4 = make_F1_confu (range - .1, tP4, 4)
    
    tConf1
    tConf2
    tConf3
    tConf4
    
  }
  
  
  
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
    
  }   #test set
  
  checking_differences = function(){
    
    setdiff( tP01$pid, tP1$pid)
    setdiff( tP1$pid, tP01$pid)
    
    tP01$dup = duplicated(tP01$pid)
    tP4$dup = duplicated(tP4$pid)
    
    setdiff(tP04$pid  ,   tP4$pid)
    setdiff(tP03$pid  ,   tP3$pid)
    
    
    setdiff(tP4$pid  ,   tP04$pid)
    setdiff(tP3$pid  ,   tP03$pid)
    setdiff(tP2$pid  ,   tP02$pid)
    
  }
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
}
