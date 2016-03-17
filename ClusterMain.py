#-*-coding:utf-8-*-
from pyspark import SparkContext
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
#writer = open('/home/yifei/TestData/data/realdata/performanceParam.csv', 'w')
#sparkContext = SparkContext()
optimalRecord = None
optimalClusterNum = None
optimalDimension = None
optimalDepth = None
optimalBin = None


def train(sparkContext):
	#truncate raw data to managable amount
	fp.truncate(pv.mergedAccountFile, pv.truncatedFile, pv.truncateLineCount)

	fp.preprocess(pv.truncatedFile, pv.processedFile, pv.targetFields)

	rawDataFrame = fp.readData(pv.trainingFile)

	simMat = cm.getSimilarityMatrixMultiProcess(rawDataFrame)

	model, unifiedRDDVecs = cluster.getClusterModel(sparkContext, simMat, rawDataFrame, pv.clusterNum, pv.dimensionReductionNum, pv.eigenVecFile)

	eva.evaluateModel(model, unifiedRDDVecs)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, pv.clusterIDCenterFile, pv.clusterIDFile)

	decisionTreeModel, trainingError, testError = classification.process(sparkContext, pv.clusterNum, pv.treeMaxDepth, pv.treeMaxBins, pv.eigenVecFile, pv.clusterIDFile)

	model.save(sparkContext, pv.clusterModelPath)

	Utils.logMessage("\nTrain cluster model finished")
	Utils.logMessage("\nCluster precision: \nTraining error %s , Test error %s" %(str(trainingError), str(testError)))

'''
def train(recordCount, clusterNum, dimension, maxDepth, maxBin):
	global writer, optimalClusterModel, optimalClassificationModel, minTrainingError, minTestError, minSSE,optimalRecord, optimalClusterNum, optimalDimension, optimalDepth, optimalBin
	
	#fp.truncate(pv.sourceFile, pv.truncatedFile, recordCount)
	fp.truncate(pv.mergedAccountFile, pv.truncatedFile, recordCount)

	fp.preprocess(pv.truncatedFile, pv.processedFile, pv.targetFields)

	rawDataFrame = fp.readData(pv.trainingFile)

	simMat = cm.getSimilarityMatrixMultiProcess(rawDataFrame)

	#sparkContext = SparkContext()

	clusterModel, unifiedRDDVecs = cluster.getClusterModel(sparkContext, simMat, rawDataFrame, clusterNum, dimension, pv.eigenVecFile)

	sse = eva.evaluateModel(clusterModel, unifiedRDDVecs)

	fp.outputNodesInSameCluster(clusterModel, unifiedRDDVecs, rawDataFrame, pv.clusterIDCenterFile, pv.clusterIDFile)

	classificationModel, trainingError, testError = classification.process(sparkContext, clusterNum, maxDepth, maxBin, pv.eigenVecFile, pv.clusterIDFile)

	writer.write("\nRun with %s record, %s clusters, %s dimensions, %s tree depth, %s bins" %(str(recordCount), str(clusterNum), str(dimension), str(maxDepth), str(maxBin)))
	writer.write("\n\tcluster sse: %s, classification training err: %s, test err: %s" %(str(sse) ,str(trainingError), str(testError)))

	if trainingError < minTrainingError and testError < minTestError:
		minTrainingError = trainingError
		minTestError = testError
		optimalClusterModel = clusterModel
		optimalClassificationModel = classificationModel
		minSSE = sse
		optimalRecord = recordCount
		optimalClusterNum = clusterNum
		optimalDimension = dimension
		optimalDepth = maxDepth
		optimalBin = maxBin

if __name__ == '__main__':
	global writer, optimalClusterModel, optimalClassificationModel, minTrainingError, minTestError, minSSE,optimalRecord, optimalClusterNum, optimalDimension, optimalDepth, optimalBin

	for recordCount in [10001]:
		for clusterNum in [300]:
			for dimension in [12]:
				for maxDepth in [5]:
					for maxBin in [32]:
						train(recordCount, clusterNum, dimension, maxDepth, maxBin)

		writer.write('\nCurrent run optimal param, record %s, cluster %s, dimension %s, maxDepth %s, maxBin %s' %(str(optimalRecord), str(optimalClusterNum), str(optimalDimension), str(optimalDepth), str(optimalBin)))
	writer.write("\n\toptimal cluster sse: " + str(minSSE) + ', classification training err: ' + str(minTrainingError) + ', test err: ' + str(minTestError))
	writer.write('\n\nFinal optimal param\nrecord %s, cluster %s, dimension %s, maxDepth %s, maxBin %s' %(str(optimalRecord), str(optimalClusterNum), str(optimalDimension), str(optimalDepth), str(optimalBin)))
	writer.close()
	optimalClusterModel.save(sparkContext, '/home/yifei/TestData/data/realdata/cluster.model')
	#optimalClassificationModel.save(sparkContext,'/home/yifei/TestData/data/realdata/classificationModel.mod')
'''
