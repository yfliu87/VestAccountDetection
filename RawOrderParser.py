import os
import codecs
import utils

def run(sourceFile, targetFile, targetFields):
	reader = codecs.open(sourceFile, 'r','gbk')
	writer = codecs.open(targetFile, 'w','gbk')

	line = reader.readline()
	fields = line.split(',')
	fieldsIdx = [] 

	for item in targetFields:
		i = fields.index(item)
		if i:
			fieldsIdx.append(i)

	while line:
		writer.write(filterFields(line, fieldsIdx))
		line = reader.readline()

	reader.close()
	writer.close()

	utils.logMessage("\nfilter fields finished")


def filterFields(line, fieldsIdx):
	fields = line.split(',')

	ret = ''
	for i in fieldsIdx:
		ret += (fields[i] + ',')

	return ret[:-1] + '\n'


def shorten(sourceFile, targetFile,):
	reader = codecs.open(sourceFile, 'r', 'gbk')
	writer = codecs.open(targetFile, 'w', 'gbk')

	count = 0
	while count < 10000:
		writer.write(reader.readline())
		count +=1

	reader.close()
	writer.close()

	utils.logMessage("\nshorten file finished")

