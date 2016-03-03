from __future__ import division
import scipy.io as scio
import pandas as pd
import Utils
from scipy import sparse
from scipy.sparse.linalg.eigen import arpack
from numpy import *
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext

def getClusters(mat, rawdata, outputFilePath,num_clusters):
	sc = SparkContext()

	laplacianMat = getLaplacianMatrix(mat)

	vals, vecs = computeEigenValsVectors(laplacianMat, num_clusters)

	unifiedRDDVecs = sc.parallelize(unification(vecs))

	model = kMeans(unifiedRDDVecs,num_clusters)

	#outputNodesInSameCluster(model, unifiedRDDVecs, rawdata, outputFilePath)

	Utils.logMessage("\nspectral cluster finished") 

	return model


def getLaplacianMatrix(mat):
	Utils.logMessage("\nconvert to Laplacian Matrix started")
	D = mat.sum(1)
	D = sqrt(1/D)
	n = len(D)
	D = D.T
	D = sparse.spdiags(D, 0, n, n)
	Utils.logMessage("\nconvert finished")

	return D * mat * D

def computeEigenValsVectors(mat, num_clusters):
	Utils.logMessage("\ncompute eigen values vectors started")

	eigenVals, eigenVecs = arpack.eigs(mat, k = num_clusters, tol=0, which = "LM")

	Utils.logMessage("\ncompute finished")

	return eigenVals, eigenVecs


def unification(vecs):
	Utils.logMessage("\nunification started")

	sq_sum = sqrt(multiply(vecs, vecs).sum(1))
	rows,cols = shape(vecs)

	for i in xrange(rows):
		for j in xrange(cols):
			vecs[i,j] = vecs[i,j]/sq_sum[i]

	Utils.logMessage("\nunification finished")
	return vecs


def kMeans(vecs, num_clusters):
	Utils.logMessage("\nkmean cluster started")

	clusters = KMeans.train(vecs, num_clusters, maxIterations=10, runs=10, initializationMode="random")

	Utils.logMessage("\nkmean cluster finished")

	return clusters


def outputNodesInSameCluster(model, unifiedRDDVecs, rawdata, target_file_path):
	Utils.logMessage("\noutput cluster started")

	df = pd.DataFrame(rawdata)
	centers = unifiedRDDVecs.map(lambda item: model.clusterCenters[model.predict(item)]).collect()
	df['center'] = centers
	df.to_csv(target_file_path, encoding='gbk', index=False)
	'''
	sorted_by_center_df = df.sort(columns='center')
	sorted_by_center_df.to_csv(target_file_path, encoding='gbk', index=False)
	'''
	Utils.logMessage("\noutput cluster finished")
