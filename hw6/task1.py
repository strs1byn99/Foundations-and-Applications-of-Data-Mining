from pyspark import SparkConf, SparkContext
import sys, time, json, random, math, copy, os
from collections import Counter, OrderedDict, defaultdict, deque
from itertools import combinations
from operator import add
import csv
import binascii

HASH_NUM = 10
NUM_OF_BIT_ARRAY = 10000
P = 4380296983 # for hash func

def hashFuncs():
    hashParams = []
    for i in range(HASH_NUM):
        a = random.randint(1, sys.maxsize)
        b = random.randint(1, sys.maxsize)
        hashParams.append((a,b))
    return hashParams

def hashIt(city, hash_params):
    # f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m
    signatures = []
    for i in range(HASH_NUM):
        a = hash_params[i][0]
        b = hash_params[i][1]
        hashed = ((a * city + b) % P) % NUM_OF_BIT_ARRAY
        signatures.append(hashed)
    return signatures

def predict(city, bloomFilter, hash_params):
    if city == None or city == "":
        return 0
    else:
        signatures = hashIt(int(binascii.hexlify(city.encode('utf8')),16), hash_params)
        if (set(signatures).issubset(bloomFilter)):
            return 1
        return 0

def main(argv):
    train_file = "business_first.json"
    in_file = "business_second.json"
    out_file = "myout"

    # train_file = argv[0]
    # in_file = argv[1]
    # out_file = argv[2]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("bloomFiltering") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    data = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: x["city"]).distinct().filter(lambda x: x != "") \
        .map(lambda x: int(binascii.hexlify(x.encode('utf8')),16))
    # print(data.count()) # 860

    # generate hash functions
    hash_params = hashFuncs()

    # generate the bloom filter
    bloomFilter = data.flatMap(lambda x: hashIt(x, hash_params)).distinct().collect()
    bloomFilter = set(bloomFilter)
    # print(bloomFilter)

    result = sc.textFile(in_file).map(lambda x: json.loads(x)).map(lambda x: x["city"]) \
        .map(lambda x: predict(x, bloomFilter, hash_params)).collect()
    # print(len(result))

    with open(out_file, "w") as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(result)

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))