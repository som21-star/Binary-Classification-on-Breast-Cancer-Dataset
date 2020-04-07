library(mlbench)
library(caret)
library(dplyr)
#Binary Classification
#Predict whether a tissue sample is malignant or benign given properties
#about the tissue sample.
data(BreastCancer)
set.seed(99)
index<-createDataPartition(BreastCancer$Class,p=0.8,list=FALSE)
train<-BreastCancer[index,]
test<-BreastCancer[-index,]

dim(BreastCancer)
glimpse(BreastCancer)
names(BreastCancer)
summary(BreastCancer)
sum(is.na(BreastCancer))
sapply(BreastCancer,class)
levels(BreastCancer$Class)

#Drop insignificant column
data<-BreastCancer[,-1]
#Convert input values into numeric
for(i in 1:9){
  data[,i]<-as.numeric(as.character(data[,i]))
}
glimpse(data)

#Dealing with missing data 
col_na<-colnames(data)[apply(data,2,anyNA)]
col_na
b_data<-data[complete.cases(data),]
str(data)
str(b_data)

#Class ditribution
perc=prop.table(table(data$Class)*100)
cbind(freq=table(data$Class), percentage=perc)
#Find correlation
cor(b_data[,1:9])

#Unimodal visualisation
par(mfrow=c(1,3))
for(i in 1:9){
  hist(b_data[,i], main=names(b_data)[i])
}
#Multimodal

#Building models
control<-trainControl(method = "repeatedcv", number = 10, repeats = 3)
#Linear combination
set.seed(99)
mod.glm<-train(Class~., b_data, method = "glm", metric = "Accuracy", trControl=control)
set.seed(99)
mod.glmnet<-train(Class~., b_data, method = "glmnet", metric = "Accuracy", trControl=control)
set.seed(99)
mod.lda<-train(Class~., b_data, method = "glm", metric = "Accuracy", trControl=control)
#nonlinear combination
set.seed(99)
mod.knn<-train(Class~., b_data, method = "knn", metric = "Accuracy", trControl=control)
set.seed(99)
mod.svm<-train(Class~., b_data, method = "svmRadial", metric = "Accuracy", trControl=control)
set.seed(99)
mod.rpart<-train(Class~., b_data, method = "rpart", metric = "Accuracy", trControl=control)
#comparison
results<-resamples(list(GLM=mod.glm, GLMNET=mod.glmnet, LDA=mod.lda, KNN=mod.knn, SVM=mod.svm, RPART=mod.rpart))
summary(results)
dotplot(results)

#Pre-processing, scaling and centering
#We will observe the result improve significantly and svm hit the top
set.seed(99)
mod.glm2<-train(Class~.,b_data,method="glm",metric="Accuracy",trControl=control,preProc=c("BoxCox"))
set.seed(99)
mod.glmnet2<-train(Class~.,b_data,method="glmnet",metric="Accuracy",trControl=control,preProc=c("BoxCox"))
set.seed(99)
mod.lda2<-train(Class~.,b_data,method="lda",metric="Accuracy",trControl=control,preProc=c("BoxCox"))
set.seed(99)
mod.knn2<-train(Class~.,b_data,method="knn",metric="Accuracy",trControl=control,preProc=c("BoxCox"))
set.seed(99)
mod.svm2<-train(Class~.,b_data,method="svmRadial",metric="Accuracy",trControl=control,preProc=c("BoxCox"))
set.seed(99)
mod.rpart2<-train(Class~.,b_data,method="rpart",metric="Accuracy",trControl=control,preProc=c("BoxCox"))

results2<-resamples(list(GLM=mod.glm2, GLMNET=mod.glmnet2, LDA=mod.lda2, KNN=mod.knn2, SVM=mod.svm2, RPART=mod.rpart2))
summary(results2)
dotplot(results2)
pred.svm<-predict(mod.svm2,b_data)
pred.svm
confusionMatrix(b_data$Class,pred.svm)
#tune parameters, another way to improve the accuracy
set.seed(99)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
mod.svm3<-train(Class~.,b_data,method="svmRadial",metric="Accuracy",trControl=control, tuneGrid=grid, preProc=c("BoxCox"))
print(mod.svm3)
plot(mod.svm3)

set.seed(99)
grid2<-expand.grid(.k=seq(1,20,by=1))
mod.knn3<-train(Class~.,b_data,method="knn",metric="Accuracy",trControl=control, tuneGrid=grid2, preProc=c("BoxCox"))
print(mod.knn3)


