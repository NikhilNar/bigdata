from collections import defaultdict
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import SQLContext
import re

# Number of most common words to remove, trying to eliminate stop words
num_of_stop_words = 50
num_topics = 3              # Number of topics we are looking for
num_words_per_topic = 10    # Number of words to display for each topic
max_iterations = 35         # Max number of times to iterate before finishing

# Initialize
conf = SparkConf().setAppName("building a warehouse")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
data = sc.wholeTextFiles('/user/ncn251/cookbook_text/*')

tokens = data \
    .map(lambda document: document.strip().lower()) \
    .map(lambda document: re.split("[\s;,#]", document)) \
    .map(lambda word: [x for x in word if x.isalpha()]) \
    .map(lambda word: [x for x in word if len(x) > 3])

# Get our vocabulary
# 1. Flat map the tokens -> Put all the words in one giant list instead of a list per document
# 2. Map each word to a tuple containing the word, and the number 1, signifying a count of 1 for that word
# 3. Reduce the tuples by key, i.e.: Merge all the tuples together by the word, summing up the counts
# 4. Reverse the tuple so that the count is first...
# 5. ...which will allow us to sort by the word count

termCounts = tokens \
    .flatMap(lambda document: document) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda tuple: (tuple[1], tuple[0])) \
    .sortByKey(False)

# Identify a threshold to remove the top words, in an effort to remove stop words
threshold_value = termCounts.take(num_of_stop_words)[num_of_stop_words - 1][0]

# Only keep words with a count less than the threshold identified above,
# and then index each one and collect them into a map
vocabulary = termCounts \
    .filter(lambda x: x[0] < threshold_value) \
    .map(lambda x: x[1]) \
    .zipWithIndex() \
    .collectAsMap()

# Convert the given document into a vector of word counts


def document_vector(document):
    id = document[1]
    counts = defaultdict(int)
    for token in document[0]:
        if token in vocabulary:
            token_id = vocabulary[token]
            counts[token_id] += 1
    counts = sorted(counts.items())
    keys = [x[0] for x in counts]
    values = [x[1] for x in counts]
    return (id, Vectors.sparse(len(vocabulary), keys, values))


# Process all of the documents into word vectors using the
# `document_vector` function defined previously
documents = tokens.zipWithIndex().map(document_vector).map(list)

# Get an inverted vocabulary, so we can look up the word by it's index value
inv_voc = {value: key for (key, value) in vocabulary.items()}

# Open an output file
with open("output.txt", 'w') as f:
    lda_model = LDA.train(documents, k=num_topics,
                          maxIterations=max_iterations)
    topic_indices = lda_model.describeTopics(
        maxTermsPerTopic=num_words_per_topic)

    # Print topics, showing the top-weighted 10 terms for each topic
    for i in range(len(topic_indices)):
        f.write("Topic #{0}\n".format(i + 1))
        for j in range(len(topic_indices[i][0])):
            f.write("{0}\t{1}\n".format(inv_voc[topic_indices[i][0][j]]
                                        .encode('utf-8'), topic_indices[i][1][j]))

    f.write("{0} topics distributed over {1} documents and {2} unique words\n"
            .format(num_topics, documents.count(), len(vocabulary)))
