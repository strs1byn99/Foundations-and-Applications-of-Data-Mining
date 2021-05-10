from pyspark import SparkConf, SparkContext, StorageLevel
import sys, time, json, random, math, string
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from operator import add
import time
import re

def calcTF(words):
    # get max kv pair based on value
    maxFrequency = max(words.items(), key = lambda x: x[1])[1]
    tf_dict = {word: (count/maxFrequency) for (word,count) in words.items()}
    return tf_dict

def aggregateProfiles(business_ids, profiles):
    l = []
    for each in business_ids:
        if (profiles.get(each) is not None):
            l.extend(profiles[each])
    return list(set(l))

def main(argv):
    in_file = "train_review.json"
    out_file = "model"
    stop_file = "stopwords"

    # in_file = argv[0]
    # out_file = argv[1]
    # stop_file = argv[2]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("task2train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    review = sc.textFile(in_file).map(lambda x: json.loads(x))
    with open(stop_file, 'r') as f_stop:
        stopwords = set(word.strip().lower() for word in f_stop)
    f_stop.close()

    ########  Data Preprocessing  ########
    toBeRemoved = set("0123456789")
    """
    (business_id, review)
    -> (business_id, combined_reviews)
    -> (business_id, [word])
    """
    concated = review.map(
        lambda x: (x["business_id"], x["text"].lower().translate({ord(ch):None for ch in toBeRemoved}))
        ).reduceByKey(lambda x,y: x + " " + y) \
        .mapValues(lambda x: re.split(r"\s|[!'\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~\\]", x)) \
        .mapValues(lambda x: [w for w in x if w not in stopwords and w != '']) \
        .persist()

    # generate rare words & remove rare words
    total_num_of_words = concated.map(lambda x: (1, len(x[1]))).reduceByKey(add).collect()[0][1]
    rare_rate = total_num_of_words * 0.000001
    rare_words = concated.flatMap(lambda x: x[1]) \
        .map(lambda x: (x, 1)).reduceByKey(add) \
        .filter(lambda x: x[0] != None and x[0] != "" and x[1] <= rare_rate).map(lambda x: x[0]).collect()
    rare_words = set(rare_words)
    concated = concated.mapValues(lambda text: [x for x in text if x not in rare_words])

    """
    -> (business_id, word)
    -> ((business_id, word), count)
    -> (business_id, (word, count))
    -> (business_id, [(word, count)])
    -> (business_id, {word: count})
    """
    business_doc = concated.flatMap(lambda x: [(x[0], w) for w in x[1]]) \
        .map(lambda x: (x, 1)).reduceByKey(add) \
        .map(lambda x: (x[0][0], (x[0][1], x[1]))) \
        .groupByKey() \
        .mapValues(lambda x: dict(x)).persist()
    concated.unpersist()
    
    ########  TF.IDF  ########
    N = business_doc.count()
    # TF = frequency of i in j / maximum occurrences of any term in document j
    # (business_id, {word: TF})
    tf = business_doc.mapValues(lambda x: calcTF(x))
    # IDF = log2(N/# of doc that mention term i)
    # (business_id, {word: count})
    # -> (word, 1) 
    # -> groupby (word, count)
    # -> (word, IDF)
    # -> {word: IDF}
    idf_dict = business_doc.flatMap(lambda x: [(word, 1) for word in x[1].keys()]) \
                .reduceByKey(add) \
                .mapValues(lambda x: math.log2(N/x)) \
                .collectAsMap()
    business_doc.unpersist()

    ########  Business Profile  ########
    # (business_id, [(word, TF.IDF)])
    # -> (business_id, [word])
    # -> {business_id: [word]}
    # -> zipWithIndex
    # -> {business_id: [word_index]}
    business_profile = tf.mapValues(lambda x: {w: t * idf_dict[w] for (w,t) in x.items()} ) \
        .mapValues(lambda x: sorted(x.items(), key=lambda kv: kv[1], reverse=True)[:200]) \
        .mapValues(lambda x: [tupl[0] for tupl in x]) 

    wordIndexes = business_profile.flatMap(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    business_profile = business_profile.mapValues(lambda x: [wordIndexes[w] for w in x]).collectAsMap()
    # print(business_profile)
    # return
    
    ########  User Profile  ########
    # (user_id, [business_id])
    # -> groupby & remove duplicate business_ids
    # -> (user_id, [word_index])
    # -> {user_id: [word_index]}
    user_profile = review.map(lambda x: (x["user_id"], x["business_id"])).groupByKey() \
                        .mapValues(lambda x: list(set(x))) \
                        .mapValues(lambda x: aggregateProfiles(x, business_profile)) \
                        .collectAsMap()
    # print(user_profile)
    
    with open(out_file, 'w') as f:
        for (k,v) in business_profile.items():
            j = {"type": "business", "business_id": k, "words": v}
            f.write(json.dumps(j))
            f.write("\n")
        for (k,v) in user_profile.items():
            j = {"type": "user", "user_id": k, "words": v}
            f.write(json.dumps(j))
            f.write("\n")
    f.close()

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))