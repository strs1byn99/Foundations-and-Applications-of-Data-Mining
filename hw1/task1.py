import sys
from pyspark import SparkContext
import json
from operator import add
from collections import OrderedDict

# A. The total number of reviews (0.5pts)
def numOfReviews(data):
    return data.count()

# B. The number of reviews in a given year, y (1pts)
def numOfReviewsInYear(data, y):
    return data.filter(lambda x: y in x['date']).count()

# C. The number of distinct users who have written the reviews (1pts)
def numOfUsers(data):
    return data.map(lambda x: x['user_id']).distinct().count()

# D. Top m users who have the largest number of reviews and its count (1pts)
def numOfTopUsers(data, m):
    users = data.map(lambda x: (x['user_id'], 1)).reduceByKey(add)
    return users.sortBy(lambda x: (-x[1], x[0])).take(int(m))

# Helper fn: remove punctuation - BUT VERY SLOW
# def punc(word, punctuation):
#     return ''.join(ch for ch in word if ch not in punctuation)

# E. Top n frequent words in the review text.
def numOfTopWords(data, n, stopwords, punctuation):
    # words = data.map(lambda x: punc(x['text'].lower(), punctuation))
    words = data.map(lambda x: x['text'].lower().translate({ord(ch):None for ch in punctuation}))
    return words.flatMap(lambda x: x.split()) \
        .filter(lambda x: x not in stopwords and x is not None and x is not "") \
        .map(lambda x: (x.strip(), 1)) \
        .reduceByKey(add) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(int(n))

def main(argv):
    in_file = argv[0]
    out_file = argv[1]
    stop_file = argv[2]
    print("in_file: " + in_file)
    y = argv[3]
    m = argv[4]
    n = argv[5]
    sc = SparkContext("local[*]", "task1")
    data = sc.textFile(in_file).map(lambda x: json.loads(x))
    with open(stop_file, 'r') as f_in:
        stopwords = set(word.strip().lower() for word in f_in)
    punctuation = set("([,.!?:;])")
    result = OrderedDict()
    result["A"] = numOfReviews(data)
    result["B"] = numOfReviewsInYear(data, y)
    result["C"] = numOfUsers(data)
    result["D"] = numOfTopUsers(data, m)
    result["E"] = [word[0] for word in numOfTopWords(data, n, stopwords, punctuation)]
    # result["E"] = numOfTopWords(data, n, stopwords, punctuation)
    print("***********************************")
    print("************ output ***************")
    print(result["A"])
    print(result["B"])
    print(result["C"])
    print(result["D"])
    print(result["E"])
    print("************ output ***************")
    print("***********************************")

    with open(out_file, 'w') as f_out:
        f_out.write(json.dumps(result))
    f_out.close()

if __name__ == '__main__':
    main(sys.argv[1:])