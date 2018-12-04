
kFoldValidation <- function (nFolds, data, modelformula,threshold) {

  # Generate array containing fold-number for each row of the dataset
  folds <- rep_len(1:nFolds, nrow(data))
  # Shuffle fold number assignment (This is equivalent to shuffeling data,
  # which ensures a good sample e.g. the original data could be initially sorted.)
  folds <- sample(folds, nrow(data))
  
  # Initialize vectors to store results
  vector_auc <- c()
  vector_acc <- c()
  vector_sen <- c()
  vector_spe <- c()

  for(k in 1:nrFolds) 
  { 
    # for each fold split data into test and training data
    fold <- which(folds == k)
    traincv <- data[-fold,]
    testcv <- data[fold,]
    # train and test model with train and test data
    glm.fit_C_cv <- glm(modelformula,
                        data = traincv,
                        family=binomial)
    result_C_cv <- predict(glm.fit_C_cv, newdata=testcv, type="response")
    # calculate AUC, accuracy, sensitivity and specificity and store results into a vector
    vector_auc <- append(vector_auc, auc(testcv$Attrition, result_C_cv))
    conf_cv <- confusionmatrix(testcv$Attrition, result_C_cv , threshold)
    vector_acc <- append(vector_acc, conf_cv[[2]])
    vector_sen <- append(vector_sen, conf_cv[[3]])
    vector_spe <- append(vector_spe, conf_cv[[4]])
  }

  return(list(vector_auc, vector_acc, vector_sen, vector_spe))
}