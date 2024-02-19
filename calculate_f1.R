# clean current envrionement variables
rm(list = ls())
# set your current working directory
setwd("/Users/ngocanh/Documents/Master/in progress/CS4220/real1"); print(getwd())

calc.F1 = function(pred,truth) {
  
  # append 'Chr' and 'StartPos'
  predv = paste(pred[,1],pred[,2])
  truthv = paste(truth[,1],truth[,2])
  
  res = data.frame(matrix(nrow = 1,ncol = 6))
  colnames(res) = c('TP','FP','FN','Precision','Recall','F1')

  res$TP = sum(predv %in% truthv)
  res$FP = sum(!(predv %in% truthv))
  res$FN = sum(!(truthv %in% predv))

  res$Precision = res$TP/(res$TP + res$FP)
  res$Recall = res$TP/(res$TP + res$FN)
  res$F1 = (2*res$Precision*res$Recall)/(res$Precision + res$Recall)
  
  return(res)
}

# load ground truth and predictions
truth = read.table('real1_truth.bed', header = F)
mypred = read.table('real1_naive1.bed', header = T)

head(truth); head(mypred)
f1stats = calc.F1(mypred,truth)

print(f1stats)