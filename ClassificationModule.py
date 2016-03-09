from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext
import pandas as pd
import Utils

def process(sc, eigenVecFile, markedClusterFile):
	eigenVec = sc.textFile(eigenVecFile)
	print eigenVec.take(3)
