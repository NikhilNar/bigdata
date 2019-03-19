from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vector, Vectors
from pyspark import SparkConf, SparkContext
import io
import zipfile
import logging

path = "/user/ncn251/cookbook_text1.zip"
totalCalls = 0


def zip_extract(x):
    totalCalls += 1
    print("zip extract called==============================", totalCalls)
    in_memory_data = io.BytesIO(x[1])
    file_obj = zipfile.ZipFile(in_memory_data, "r")
    files = [i for i in file_obj.namelist()]
    return [file_obj.open(file).read() for file in files]


conf = SparkConf().setAppName("lda")
sc = SparkContext(conf=conf)
zips = sc.binaryFiles(path, 100)
zipData = sc.parallelize(zips.map(zip_extract).collect())
print("zipData====================", zipData.count())
data = zipData.zipWithIndex().map(lambda words: Row(
    idd=words[1], words=words[0].split(" ")))

logging.info("------------------------data read successfully--------------------------------------------------------------------")

docDF = SQLContext(sc).createDataFrame(data)
Vector = CountVectorizer(inputCol="words", outputCol="vectors")
model = Vector.fit(docDF)
result = model.transform(docDF)

corpus = result.select("idd", "vectors").rdd.map(
    lambda x: [x[0], Vectors.fromML(x[1])]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3, maxIterations=100, optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary


wordNumbers = 100  # number of words per topic
topicIndices = sc.parallelize(
    ldaModel.describeTopics(maxTermsPerTopic=wordNumbers))


def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
    print("Topic" + str(topic) + ":")
    print(topics_final[topic])
#     for term in topics_final[topic]:
#         print (term)
    print('\n')
