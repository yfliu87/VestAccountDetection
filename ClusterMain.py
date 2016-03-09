#-*-coding:utf-8-*-
from pyspark import SparkContext
import ClusterModule as cluster
import FileParser as fp
import ComputeModule as cm
import PredefinedValues as pv
import Evaluation as eva
import ClassificationModule as classification

def run():
	#truncate raw data to managable amount
	fp.truncate(pv.sourceFile, pv.truncatedFile, pv.truncateLineCount)

	fp.preprocess(pv.truncatedFile, pv.processedFile, pv.targetFields)

	rawDataFrame = fp.readData(pv.trainingFile)

	simMat = cm.getSimilarityMatrixMultiProcess(rawDataFrame)

	sparkContext = SparkContext()

	model, unifiedRDDVecs = cluster.getClusterModel(sparkContext, simMat, rawDataFrame, pv.clusterNum, pv.dimensionReductionNum, pv.eigenVecFile)

	eva.evaluateModel(model, unifiedRDDVecs)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, pv.clusterIDCenterFile, pv.clusterIDFile)

	classification.process(sparkContext, pv.clusterNum, pv.eigenVecFile, pv.clusterIDFile)


if __name__ == '__main__':
	run()
	
