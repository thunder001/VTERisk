
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
make_P = function(Tr){
  data.frame(pid = Tr$patient_id, 
             p = Tr$probs, y = ifelse(Tr$golds=="True", 1, 0)  )
  
}

CSV_Path = 'G:/FillmoreCancerData/markhe/VTERisk/Notebooks/Calibrate/R/CSV_for_R/'

Get_4_I = function(Path){

  M = read.csv(Path)
  M$X = NULL
  Ms = split_w_rep(M)
  Y1 = Ms$Y1
  Y2 = Ms$Y2
  Y3 = Ms$Y3
  Y4 = Ms$Y4 
  return(list(P1 = make_P(Y1),
              P2 = make_P(Y2),
              P3 = make_P(Y3),
              P4 = make_P(Y4)))
}
 
Tr.M = Get_4_I(paste0 (CSV_Path, 'Train_1_27.csv'))
Mdevs = Get_4_I(paste0 (CSV_Path, 'Dev_1_27.csv'))
Mtests = Get_4_I(paste0 (CSV_Path, 'Test_1_27.csv'))

P1 = Tr.M$P1
P2 = Tr.M$P2
P3 = Tr.M$P3
P4 = Tr.M$P4

R_tP01 =  (Mtests$P1)
R_tP02 =  (Mtests$P2)
R_tP03 =  (Mtests$P3)
R_tP04 =  (Mtests$P4)

Pdev1 =  (Mdevs$P1)
Pdev2 =  (Mdevs$P2)
Pdev3 =  (Mdevs$P3)
Pdev4 =  (Mdevs$P4)


# need to visually adjust and tweak parameters of spline

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
Dev3_balanced2 = redo_frequency_on_dev(Pdev3, P3,d1 = 4, d2 = -1, eps = -2)
Dev4_balanced2 = redo_frequency_on_dev(Pdev4, P4,d1 = 3, d2 = 2, eps = -2.3)

#Dev4_balanced2 = redo_frequency_on_dev(Pdev4, P4,d1 = 3, d2 = -.5, eps = -1)

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

cv2 = spline_pred (Dev2_balanced2, R_tP02,  knots = 14, pen =  .1) 
P.cv2 =  data.frame(p = as.numeric(cv2), y  =  R_tP02$y  )
tCal2  = calibration_plot(data = P.cv2, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05), 
                          xlab = "Binned Pred (Interval 2: 3-6 Months after Index Date)")

cv3 = spline_pred (Dev3_balanced2, R_tP03) 
P.cv3 =  data.frame(p = as.numeric(cv3), y  =  R_tP03$y  )
tCal3  = calibration_plot(data = P.cv3, obs = 'y', pred = 'p', x_lim = c(0,.05), y_lim = c(0,.05),
                          xlab = "Binned Pred (Interval 3: 6-9 Months after Index Date)")

cv4 = spline_pred (Dev4_balanced2, R_tP04, knots = 11, pen = .5) 
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
 