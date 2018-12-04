rocplot <- function(pred,truth,...) {
  predob = prediction(pred,truth)
  perf = performance(predob,"tpr","fpr")
  plot(perf,...)
}

confusionmatrix <- function(testdata, modelpred, threshold){
  confmatrix <- table(testdata, modelpred >= threshold)
  accuracy <- (confmatrix[1,1] + confmatrix[2,2]) / length(testdata)
  sensitivity <- confmatrix[2,2] / (confmatrix[2,2] + confmatrix[2,1]) 
  specificity <- confmatrix[1,1] / (confmatrix[1,1] + confmatrix[1,2]) 
  return (list(confmatrix, accuracy, sensitivity, specificity))
}
