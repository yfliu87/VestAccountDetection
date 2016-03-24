#-*-coding:utf-8-*-
targetFields = ['order_id','buyer_pin','buyer_full_name','buyer_full_address','buyer_mobile','buyer_ip','equipment_id','buyer_city_name','buyer_country_name','buyer_poi','promotion_id','label']
fileColumns = ['buyer_pin','equipment_id','buyer_ip','buyer_poi','promotion_id']
simWeight = {'buyer_ip':20, 
			 'buyer_mobile':0, 
			 'buyer_full_address':0, 
			 'equipment_id': 30, 
			 'buyer_full_name':0,
			 'buyer_city_name':0,
			 'buyer_county_name':0,
			 'buyer_poi':30,
			 'promotion_id':20}

DEFAULTSIM = 0.0

truncatedFile = '/root/TestData/realdata/truncated.csv'
processedFile = '/root/TestData/realdata/processed.csv'
trainingFile = processedFile
testFile = ''
eigenVecFile = '/root/TestData/realdata/eigenVec.csv'
clusterIDCenterFile = '/root/TestData/realdata/clustered_id_center.csv'
clusterIDFile = '/root/TestData/realdata/clustered_id.csv'
IDFOREACHCLUSTER = 20
truncateLineCount = 1001
clusterNum = truncateLineCount/IDFOREACHCLUSTER
dimensionReductionNum = 10
treeMaxDepth = 4
treeMaxBins = 16

#below parameters only for classification model
confirmedAccountFile = '/root/TestData/realdata/ruleFilteredAccount.txt'
randomAccountFile = '/root/TestData/realdata/randomAccount.txt' 
rule1AccountFile = '/root/TestData/realdata/output_rule1.txt' 
rule2AccountFile = '/root/TestData/realdata/output_rule2.txt' 
rule3AccountFile = '/root/TestData/realdata/output_rule3.txt' 
rule4AccountFile = '/root/TestData/realdata/output_rule4.txt' 
rule5AccountFile = '/root/TestData/realdata/output_rule5.txt' 
rule6AccountFile = '/root/TestData/realdata/output_rule6.txt' 
mergedAccountFile = '/root/TestData/realdata/mergedAccount.txt' 
fileForClusterModel = '/root/TestData/realdata/clusterRawData.txt'

#below params are defined for mixed model demo
classificationModelPath = '/root/TestData/realdata/classification.model'
clusterModelPath = '/root/TestData/realdata/cluster.model'
simMatrixFile = '/root/TestData/realdata/simMat.txt'
isTrainingRound = True
outputDebugMsg = False
isLastRecord = False
simThreshold = 50