from pyspark import SparkContext
import sys, time, json, random, math
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from operator import add
import time

def hashFuncs(hashNum):
    hashParams = []
    for i in range(hashNum):
        a = random.randint(1, sys.maxsize)
        b = random.randint(1, sys.maxsize)
        hashParams.append((a,b))
    return hashParams

def hashIt(hashNum, bucketNum, toBeHashed, hashParams):
    signatures = []
    for i in range(hashNum):
        a = hashParams[i][0]
        b = hashParams[i][1]
        hashedValues = []
        for x in toBeHashed:
            hashed = ((a * x + b) % 4380296983) % bucketNum
            hashedValues.append(hashed)
        signatures.append(min(hashedValues))
    return signatures

def splitBands(signts, bandNum):
    bands = []
    row = math.ceil(len(signts) / bandNum)
    for i in range(0, len(signts)):
        bands.append((i, hash(tuple(signts[i*row:(i+1)*row]))))
    # [(band_id, hash of partial signatures)]
    return bands

def jaccard(t, matrix):
    b1 = t[0]
    b2 = t[1]
    union = set(matrix[b1]).union(set(matrix[b2]))
    intersection = set(matrix[b1]).intersection(matrix[b2])
    return len(intersection) / len(union)

def main(argv):
    in_file = "train_review.json"
    out_file = "myout"

    # in_file = argv[0]
    # out_file = argv[1]
    
    hashNum = 70
    bandNum = hashNum

    sc = SparkContext(master="local[*]", appName="task1")
    review = sc.textFile(in_file).map(lambda x: json.loads(x))
    
    # (user_id, [business_id])
    user_business = review.map(lambda x: (x["user_id"], x["business_id"])).groupByKey()
    # (old_user_idx, [business_id])
    indexed = user_business.zipWithIndex().map(lambda x: (x[1], x[0][1]))
    # flatMap -> (business_id, old_user_idx) 
    # -> (business_id, [old_user_idx])
    characteristic_matrix = indexed.flatMap(lambda x: [(b, x[0]) for b in x[1]]).groupByKey()
    
    #########  MinHash  #########
    # generate hash functions
    m = indexed.count()
    hash_params = hashFuncs(hashNum) # a,b parameter in each hash func
    # -> (business_id, [h1, h2, h3...])
    signature_matrix = characteristic_matrix.map(lambda x: (x[0], hashIt(hashNum, m, x[1], hash_params)))
    # print(signature_matrix.collect())

    #########  Locality Sensitive Hashing  #########
    # -> ((band_id, hash), business_id)
    # -> ((band_id, hash), [business_id])
    # -> (b1, b2)
    candidates = signature_matrix.flatMap(lambda x: [(band, x[0]) for band in splitBands(x[1], bandNum)]) \
        .groupByKey() \
        .mapValues(lambda x: list(set(x))).filter(lambda x: len(x) > 1) \
        .flatMap(lambda x: list(combinations(x[1], 2))).distinct().map(lambda x: tuple(sorted(x)))

    # Jaccard Similarity
    cMatrix = characteristic_matrix.mapValues(lambda x: list(x)).collectAsMap()
    # print(cMatrix)
    # -> ((b1, b2), similarity)
    results = candidates.map(lambda x: (x, jaccard(x, cMatrix))).filter(lambda x: x[1] >= 0.05).collect()
    #print(results)

    with open(out_file, 'w') as f:
        for each in results:
            f.write(json.dumps({"b1": each[0][0], "b2": each[0][1], "sim": each[1]}) + "\n")


if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start)) 