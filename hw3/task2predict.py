from pyspark import SparkConf, SparkContext
import sys, time, json, random, math, string
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from operator import add
import time

def cosine(x):
    uprofile = set(x[0])
    bprofile = set(x[1])
    if len(uprofile) != 0 and len(bprofile) != 0:
        return len(uprofile.intersection(bprofile)) / math.sqrt(len(uprofile) * len(bprofile))
    else: 
        return 0

def main(argv):
    in_file = "test_review.json"
    model_file = "model"
    out_file = "myout2"

    # in_file = argv[0]
    # model_file = argv[1]
    # out_file = argv[2]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("task2predict") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    model = sc.textFile(model_file).map(lambda x: json.loads(x))
    user_profile = model.filter(lambda x: x["type"] == "user") \
                    .map(lambda x: (x["user_id"], x["words"])) \
                    .collectAsMap()
    business_profile = model.filter(lambda x: x["type"] == "business") \
                        .map(lambda x: (x["business_id"], x["words"])) \
                        .collectAsMap()
    
    # ([user_word], [business_word])
    # -> (([user_word], [business_word]), sim)
    data = sc.textFile(in_file).map(lambda x: json.loads(x)) \
            .map(lambda x: (x["user_id"], x["business_id"])) \
            .map(lambda x: ((x[0], x[1]), (user_profile.get(x[0]), business_profile.get(x[1])))) \
            .filter(lambda x: x[1][0] != None and x[1][1] != None) \
            .map(lambda x: (x[0], cosine(x[1]))) \
            .filter(lambda x: x[1] >= 0.01) \
            .collect()
    # print(data)
 
    with open(out_file, 'w') as f:
        for each in data:
            f.write(json.dumps({"user_id": each[0][0], "business_id": each[0][1], "sim": each[1]}) + "\n")

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))