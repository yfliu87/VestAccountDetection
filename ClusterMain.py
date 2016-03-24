#-*-coding:utf-8-*-
from pyspark import SparkContext
import pandas as pd
import ClusterModule as cluster
import FileParser as fp
import ComputeModule as cm
import PredefinedValues as pv
import Evaluation as eva
import ClassificationModule as classification
import Utils

optimalClusterModel = None
optimalClassificationModel = None
minTrainingError = 1.0
minTestError = 1.0
minSSE = 1.0
optimalRecord = None
optimalClusterNum = None
optimalDimension = None
optimalDepth = None
optimalBin = None


def calculateSimMat():
	fp.truncate(pv.mergedAccountFile, pv.truncatedFile, pv.truncateLineCount)
	fp.preprocess(pv.truncatedFile, pv.processedFile, pv.targetFields)
	pd.read_csv(pv.trainingFile, sep=',',encoding='utf-8').to_csv(pv.fileForClusterModel, index=False, encoding='utf-8')
	rawDataFrame = pd.read_csv(pv.fileForClusterModel, sep=',',encoding='utf-8')
	simMat = cm.getSimilarityMatrix(sparkContext, rawDataFrame)
	fp.recordSimMatrix(simMat, pv.simMatrixFile)


def train(sparkContext):
	fp.truncate(pv.mergedAccountFile, pv.truncatedFile, pv.truncateLineCount)

	fp.preprocess(pv.truncatedFile, pv.processedFile, pv.targetFields)

	pd.read_csv(pv.trainingFile, sep=',',encoding='utf-8').to_csv(pv.fileForClusterModel, index=False, encoding='utf-8')

	rawDataFrame = pd.read_csv(pv.fileForClusterModel, sep=',',encoding='utf-8')

	simMat = cm.getSimilarityMatrix(sparkContext, rawDataFrame)

	fp.recordSimMatrix(simMat, pv.simMatrixFile)

	model, unifiedRDDVecs = cluster.getClusterModel(sparkContext, simMat, rawDataFrame, (pv.truncateLineCount/pv.IDFOREACHCLUSTER), pv.dimensionReductionNum, pv.eigenVecFile)

	eva.evaluateModel(model, unifiedRDDVecs)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, pv.clusterIDCenterFile, pv.clusterIDFile)

	decisionTreeModel = classification.process(sparkContext, (pv.truncateLineCount/pv.IDFOREACHCLUSTER), pv.treeMaxDepth, pv.treeMaxBins, pv.eigenVecFile, pv.clusterIDFile)

	model.save(sparkContext, pv.clusterModelPath)

	Utils.logMessage("\nTrain cluster model finished")


if __name__ == '__main__':
	pass
