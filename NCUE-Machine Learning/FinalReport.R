library(class);library(caret);library(pROC);library(e1071);library(nnet)
library(sampling); library(dplyr); library(tree);library(randomForest)
#載入資料、資料型態設定----
AllData=read.csv("diabetes.csv",header=T)
AllData$class=as.factor(AllData$class)
AllData$Obesity=as.factor(AllData$Obesity)
AllData$Alopecia=as.factor(AllData$Alopecia)
AllData$muscle_stiffness=as.factor(AllData$muscle_stiffness)
AllData$partial_paresis=as.factor(AllData$partial_paresis)
AllData$delayed_healing=as.factor(AllData$delayed_healing)
AllData$Irritability=as.factor(AllData$Irritability)
AllData$Itching=as.factor(AllData$Itching)
AllData$visual_blurring=as.factor(AllData$visual_blurring)
AllData$Genital_thrush=as.factor(AllData$Genital_thrush)
AllData$Polyphagia=as.factor(AllData$Polyphagia)
AllData$weakness=as.factor(AllData$weakness)
AllData$sudden_weight_loss=as.factor(AllData$sudden_weight_loss)
AllData$Polydipsia=as.factor(AllData$Polydipsia)
AllData$Polyuria=as.factor(AllData$Polyuria)
AllData$Gender=as.factor(AllData$Gender)
AllN=nrow(AllData)
#所有資料的10%當作testing set，剩下的90%當成available set
contrasts(AllData$class)
GN.test=round(table(AllData$class)*0.1,0)
set.seed(3)
Testget=strata(AllData,"class",size=c(GN.test[[1]],GN.test[[2]]),method="srswor")
TestInx=Testget$ID_unit
TestData= AllData[TestInx,]
AvaInx=c(1:AllN)[-TestInx]
AvaData=AllData[AvaInx,]

##Classification tree----
#由available set建構classification tree並畫出
ModelTree=tree(class~., data=AvaData)
ModelTree
win.graph();plot(ModelTree);text(ModelTree)
#由10-fold CV畫出tree size vs.模型error rate的關係來判斷是否prune tree
#有最小error rate的模型，為最適tree size
set.seed(5)
ModelTreeCV=cv.tree(ModelTree, FUN=prune.misclass)
win.graph()
plot(ModelTreeCV$size, ModelTreeCV$dev,type="b")

ModelTreeCV$size;ModelTreeCV$dev

#挑選最適tree size修剪樹並畫出
ModelPruneTree=prune.tree(ModelTree,best=5)
win.graph()
plot(ModelPruneTree)
text(ModelPruneTree)

#將testing data帶入修剪後的樹，得到Y預測值計算Accuracy
PredProbY=predict(ModelPruneTree,TestData[,-17])
PredY=as.factor(ifelse(as.data.frame(PredProbY)$Positive>0.5,"Positive","Negative"))
confusionMatrix(PredY, TestData$class, positive='Positive' )
#畫ROC curve & 計算AUC
win.graph()
plot.roc(as.numeric(TestData$class),as.numeric(PredY), print.auc=TRUE)


##Logistic+validation set approach----
#80%training set、20#validation set
GN.train=round(table(AvaData$class)*0.8,0)
set.seed(5)
Trainget=strata(AvaData,"class",size=c(GN.train[[1]],GN.train[[2]]),method="srswr")
TrainData=getdata(AvaData,Trainget)
TrainInx=Trainget$ID_unit
TrainData=AvaData[TrainInx,]
ValInx=c(1:nrow(AvaData))[-TrainInx]
ValData=AvaData[ValInx,]

#Model 1 Full model
ModelLog1=glm(formula=class~.,family=binomial,data=TrainData)
PreProb1=predict(ModelLog1, newdata=ValData[,-17],type="response")
PredY1=as.factor(ifelse(PreProb1>0.5, "Positive", "Negative"))#機率大於0.5被分到SO否則R
confusionMatrix(PredY1, ValData$class)
win.graph()
plot.roc(as.numeric(ValData$class),as.numeric(PredY1),print.auc=TRUE)
auc(roc(as.numeric(ValData$class),as.numeric(PredY1)))
#Model 2
ModelLog2=glm(formula=class~Polydipsia,family=binomial,data=TrainData)
PreProb2=predict(ModelLog2, newdata=ValData[,-17],type="response")
PredY2=as.factor(ifelse(PreProb2>0.5, "Positive", "Negative"))
confusionMatrix(PredY2, ValData$class, positive='Positive' )
auc(roc(as.numeric(ValData$class),as.numeric(PredY2)))
#model 3
ModelLog3=glm(formula=class~Polydipsia+Polyuria,family=binomial,data=TrainData)
PreProb3=predict(ModelLog3, newdata=ValData[,-17],type="response")
PredY3=as.factor(ifelse(PreProb3>0.5, "Positive", "Negative"))
confusionMatrix(PredY3, ValData$class)
auc(roc(as.numeric(ValData$class),as.numeric(PredY3)))
#model 4
ModelLog4=glm(formula=class~Polydipsia+Polyuria+Gender,family=binomial,data=TrainData)
PreProb4=predict(ModelLog4, newdata=ValData[,-17],type="response")
PredY4=as.factor(ifelse(PreProb4>0.5, "Positive", "Negative"))
confusionMatrix(PredY4, ValData$class)
auc(roc(as.numeric(ValData$class),as.numeric(PredY4)))
#model 5
ModelLog5=glm(formula=class~Polydipsia+Polyuria+Gender+Alopecia,family=binomial,data=TrainData)
PreProb5=predict(ModelLog5, newdata=ValData[,-17],type="response")
PredY5=as.factor(ifelse(PreProb5>0.5, "Positive", "Negative"))
confusionMatrix(PredY5, ValData$class)
auc(roc(as.numeric(ValData$class),as.numeric(PredY5)))
#找出最適模型(以accuracy為準則)
Accu=rep(NA,5)
Accu[1]=confusionMatrix(PredY1,ValData$class)$overall["Accuracy"]
Accu[2]=confusionMatrix(PredY2,ValData$class)$overall["Accuracy"]
Accu[3]=confusionMatrix(PredY3,ValData$class)$overall["Accuracy"]
Accu[4]=confusionMatrix(PredY4,ValData$class)$overall["Accuracy"]
Accu[5]=confusionMatrix(PredY5,ValData$class)$overall["Accuracy"]
which.max(Accu)#所有model中validation accuracy最大的為最適模型#1最大

#以availabla set建構最適模型
FinalModel=glm(formula=class~.,family=binomial,data=AvaData)
summary(FinalModel)
#計算testing set的結果
TestPreProb=predict(FinalModel, newdata=TestData[,-17], type="response")
TestPredY=as.factor(ifelse(TestPreProb>0.5, "Positive", "Negative"))
confusionMatrix(TestPredY, TestData$class, positive='Positive')
win.graph()
plot.roc(as.numeric(TestData$class),as.numeric(TestPredY),print.auc=TRUE)
auc(roc(as.numeric(TestData$class),as.numeric(TestPredY)))

#找出最適模型(以AUC為準則)
AUC=rep(NA,5)
AUC[1]=auc(roc(as.numeric(ValData$class),as.numeric(PredY1)))
AUC[2]=auc(roc(as.numeric(ValData$class),as.numeric(PredY2)))
AUC[3]=auc(roc(as.numeric(ValData$class),as.numeric(PredY3)))
AUC[4]=auc(roc(as.numeric(ValData$class),as.numeric(PredY4)))
AUC[5]=auc(roc(as.numeric(ValData$class),as.numeric(PredY5)))
which.max(AUC)#所有model中validation AUC最大的為最適模型

#以availabla set建構最適模型
FinalModel=glm(formula=class~.,family=binomial,data=AvaData)
summary(FinalModel)
#計算testing set的結果
TestPreProb=predict(FinalModel, newdata=TestData[,-17], type="response")
TestPredY=as.factor(ifelse(TestPreProb>0.5, "Positive", "Negative"))
confusionMatrix(TestPredY, TestData$class, positive='Positive')
auc(roc(as.numeric(TestData$class),as.numeric(TestPredY)))


##Bagging----
#由Available set建構bagging model
set.seed(2)
ModelBag=randomForest(class~., data=AvaData,mtry=10, importance=T)
#觀察各X的重要性
importance(ModelBag)
varImpPlot(ModelBag)
#由Testing set的X帶入bagging model預測class，並計算準確率
PredY=predict(ModelBag,newdata=TestData[,-17],type="response")
confusionMatrix(PredY, TestData$class, positive='Positive' )
#畫出ROC curve&計算AUC
win.graph()
plot.roc(as.numeric(TestData$class),as.numeric(PredY), print.auc=TRUE)

##Random forest----
#由available set建構random forest model(m=√p)
set.seed(2)
ModelRF=randomForest(class~., data=AvaData,mtry=3, importance=T)
#觀察各X的重要性
round(importance(ModelRF),2)
varImpPlot(ModelRF)

#由Testing set的X帶入bagging model預測class，並計算準確率
PredY=predict(ModelRF,newdata=TestData[,-17],type="response")
confusionMatrix(PredY, TestData$class, positive='Positive' )
#Out-of-bag error Estimation
win.graph()
plot(ModelRF)

#畫出ROC curve&計算AUC
win.graph()
plot.roc(as.numeric(TestData$class),as.numeric(PredY), print.auc=TRUE)

##Support vector machine+validation set approach----
#Available set當中的80%，當training set，剩下的20%為validation set
GN.train=round(table(AvaData$class)*0.8,0)
set.seed(5)
Trainget=strata(AvaData,"class",size=c(GN.train[[1]],GN.train[[2]]),method="srswr")
TrainData=getdata(AvaData,Trainget)
TrainInx=Trainget$ID_unit
TrainData=AvaData[TrainInx,]
ValInx=c(1:nrow(AvaData))[-TrainInx]
ValData=AvaData[ValInx,]

#設定各種cost(tuning parameter)的值
Costlist=c(0.001,0.01,0.1,1,5,10,100)
AccuSumm=rep(0,length(Costlist))

#對每個tuning parameter值，以training set建構model
#並以validation set的X，代入建構的model，得到Y的預測情況並計算準確率
for(i in 1:length(Costlist)){
  svmfitTemp=svm(class~., data=TrainData,kernel="linear",cost=Costlist[i], scale=F)
  PredYTemp=predict(svmfitTemp, newdata=ValData[,-17],type="response")
  AccuSumm[i]=confusionMatrix(PredYTemp,ValData$class)$overall["Accuracy"]
}

#找best tuning parameter value (準確率最高)
BestC=which.max(AccuSumm)
#由available set 和 best tuning parameter value建構最終模型
ModelLinearSVM=svm(class~., data=AvaData,kernel="linear",cost=Costlist[BestC], scale=F)
summary(ModelLinearSVM)
ModelLinearSVM$index

#將testing data的X帶入finalmodel，得到Y預測值並計算accuracy
PredY=predict(ModelLinearSVM, newdata=TestData[,-17],type="response")
confusionMatrix(PredY, TestData$class, positive='Positive' )
#畫ROC curve & 計算AUC
win.graph()
plot.roc(as.numeric(TestData$class),as.numeric(PredY),print.auc=TRUE)


##Support vector machine+10-Fold CV----
#設定各種cost(tuning parameter)的值
Costlist=c(0.001,0.01,0.1,1,5,10,100)
#利用available set data，由10-fold CV計算各種tuning parameter value時的準確率
set.seed(3)
tune.out=tune(svm,class~.,data=AvaData, kernel="linear",ranges=list(cost=Costlist) )
summary(tune.out)
#準確率最高者為最佳模型
bestmod=tune.out$best.model
summary(bestmod)
#將testing data的X帶入final model，得到Y預測值並計算accuracy
PredY=predict(bestmod, newdata=TestData[,-17],type="response")
confusionMatrix(PredY, TestData$class, positive='Positive' )
#畫ROC curve & 計算AUC
win.graph()
plot.roc(as.numeric(TestData$class),as.numeric(PredY),print.auc=TRUE)


##Neural Network----
#Available set當中的80%，當training set，剩下的20%為validation set
GN.train=round(table(AvaData$class)*0.8,0)
set.seed(5)
Trainget=strata(AvaData,"class",size=c(GN.train[[1]],GN.train[[2]]),method="srswr")
TrainData=getdata(AvaData,Trainget)
TrainInx=Trainget$ID_unit
TrainData=AvaData[TrainInx,]
ValInx=c(1:nrow(AvaData))[-TrainInx]
ValData=AvaData[ValInx,]

#原本可由training set建構neural network model
#並由validation set代入建構的模型計算準確率，找最適的超參數
NNfit=nnet(class~., data=AvaData, size=50, range=0.7)
summary(NNfit)
NNfit$wts ##所有weight的係數估計值
#將testing data的X帶入final model，得到Y預測值並計算accuracy
PredY=predict(NNfit,newdata=TestData[,-17],type="class")
confusionMatrix(as.factor(PredY), TestData$class, positive='Positive' )
#畫ROC curve & 計算AUC
win.graph()
plot.roc(as.numeric(TestData$class), as.numeric(as.factor(PredY)), print.auc=TRUE)


##Plot----
#Age
divide_age=cut(AllData$Age, c( 15, 21, 39, 69, 90),)
win.graph()
ggplot(data = AllData, aes(x = divide_age, fill = class)) +
  geom_bar(position = "stack", width=0.5)

win.graph()
ggplot(data = AllData, aes(x = Gender, fill = class)) +
  geom_bar(position = "stack", width=0.7)

win.graph()
ggplot(data = AllData, aes(x = Polyuria, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Polydipsia, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = sudden_weight_loss, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = weakness, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Polyphagia, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Genital_thrush, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = visual_blurring, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Itching, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Irritability, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = delayed_healing, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = partial_paresis, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = muscle_stiffness, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Alopecia, fill = class)) +
  geom_bar(position = "stack", width=0.55)

win.graph()
ggplot(data = AllData, aes(x = Obesity, fill = class)) +
  geom_bar(position = "stack", width=0.55)
