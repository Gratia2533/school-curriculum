AvaData=read.csv("Car_Purchasing_data.csv",header=T,row.names = NULL)
AvaN=nrow(AvaData)
AvaData$Gender=as.factor(AvaData$Gender)#將性別轉為類別變數，0男1女

#Validation set approach
#80%資料作為training set/20%作為validation set
TrainN=nrow(AvaData)*0.8
set.seed(1)
TrainInx=sample(c(1:AvaN),TrainN)
ValInx=c(1:AvaN)[-TrainInx]
TrainData=AvaData[TrainInx,]
ValData=AvaData[ValInx,]

#用training set建構迴歸模型
library(leaps)
BestSubModel=regsubsets(Car_Purchase_Amount~., data=TrainData, nvmax=5)
summary(BestSubModel)
#由validation set分別計算d=1,2,...,5最佳模型的validation MSE
ValDataM=model.matrix(Car_Purchase_Amount~., data=ValData)
ValMSE=rep(NA,5)
for(d in 1:5){
  coef.d<-coef(BestSubModel,d)
  predY<-ValDataM[,names(coef.d)]%*%coef.d
  ValMSE[d]<-mean((ValData$Car_Purchase_Amount-predY)^2)}
#找出MSE最小者，挑選最適合的p
BestD=which.min(ValMSE)
BestD
#由Available set建構迴歸模型
BestModel=regsubsets(Car_Purchase_Amount~., data=AvaData,nvmax=7)
coef(BestModel, BestD)
RegModel.F=lm(Car_Purchase_Amount~Gender+Age+Annual_Salary+
                Credit_Card_Debt+Net_Worth,data=AvaData)


#Create 10-Fold
library(caret)
FoldK=10#設定K=10
ValMSE_K=matrix(NA,FoldK,5)
set.seed(1)
CFold=createFolds(c(1:AvaN),k=FoldK,returnTrain=T)
#由training set建構迴歸模型
#由validation set計算d=1,2..,5模型的validation MSE
for(j in 1:FoldK){
  TrainInxTemp=CFold[[j]]
  ValInxTemp=c(1:AvaN)[-TrainInxTemp]
  TrainDataTemp=AvaData[TrainInxTemp,]
  ValDataTemp=AvaData[ValInxTemp,]
  BestSubModel_K=regsubsets(Car_Purchase_Amount~., data=TrainDataTemp,nvmax=5)
  ValDataM_K=model.matrix(Car_Purchase_Amount~., data=ValDataTemp)
  for(d in 1:5){
    coef.d<-coef(BestSubModel_K,d)
    predY<-ValDataM_K[,names(coef.d)]%*%coef.d
    ValMSE_K[j,d]<-mean((ValDataTemp$Car_Purchase_Amount-predY)^2)
  }
}
ValMSE_K
#用10 folds的MSE平均作為validation MSE
MeanD=colMeans(ValMSE_K)
SDD=apply((ValMSE_K),FUN=sd, MARGIN=2)
#由Validation MSE最小者挑出最適合的d
summary(BestSubModel_K)
BestD_K=which.min(colMeans(ValMSE_K))
BestModel_K=regsubsets(Car_Purchase_Amount~., data=AvaData, nvmax=5)
coef(BestModel_K, BestD_K)
RegModel_K.F=lm(Car_Purchase_Amount~Age+Annual_Salary+Net_Worth,data=AvaData)

#模型準確程度
plot(predict(RegModel_K.F), AvaData$Car_Purchase_Amount
     ,xlab="Predicted values",ylab = "Actual values",pch=20)
abline(a = 0,b = 1,col = "red",lwd = 2)

#Fitted vs. residual #Normal QQ-plot
plot(RegModel_K.F)
#install.packages("car")
library(car)
durbinWatsonTest(RegModel_K.F)
ncvTest(RegModel_K.F)

#由圖形檢查outlier/influential point/high leverage point

influencePlot(RegModel_K.F)
leveragePlots(RegModel_K.F)
influenceIndexPlot(RegModel_K.F)
outlierTest(RegModel_K.F)

vif(RegModel_K.F)#檢查迴歸的collinearity



#Gender
plot(xlab="Gender",ylab="Car_Purchase_Amount", AvaData$Gender, 
     AvaData$Car_Purchase_Amount, pch=19, col="#336666")
#Age
plot(AvaData$Age, AvaData$Car_Purchase_Amount, pch=20, col="#5151A2")
lm.model_Age <- lm(Car_Purchase_Amount~Age, AvaData)
abline(lm.model_Age,lwd=2)
#Annual_Salary
plot(AvaData$Annual_Salary, AvaData$Car_Purchase_Amount, pch=20, col="#A23400")
lm.model_AS <- lm(Car_Purchase_Amount~Annual_Salary, AvaData)
abline(lm.model_AS,lwd=2)
#Credit_Card_Debt
plot(AvaData$Credit_Card_Debt, AvaData$Car_Purchase_Amount, pch=20, col="#0072E3")
lm.model_CCD <- lm(Car_Purchase_Amount~Credit_Card_Debt, AvaData)
abline(lm.model_CCD,lwd=2)
#Net_Worth
plot(AvaData$Net_Worth, AvaData$Car_Purchase_Amount, pch=20, col="#4F9D9D")
lm.model_NW <- lm(Car_Purchase_Amount~Net_Worth, AvaData)
abline(lm.model_NW,lwd=2)







