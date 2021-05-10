from pyspark import SparkConf, SparkContext
import sys, time, json, random, math
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from operator import add
import time

CO_RATED_NUM = 3
hashNum = 40
bandNum = hashNum

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
            hashed = (a * x + b) % bucketNum
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

def filterByCoRatedNum(b1, b2):
    return True if len(set(b1.keys()).intersection(set(b2.keys()))) >= CO_RATED_NUM else False

def filterByCoRatedNum1(u1, u2):
    return True if len(set(u1).intersection(set(u2))) >= CO_RATED_NUM else False

def pearson(b1, b2):
    # list of co-rated users
    coRatedUser = set(b1.keys()).intersection(set(b2.keys()))
    coRatedUserOfB1, coRatedUserOfB2 = list(),list()
    [(coRatedUserOfB1.append(b1[user]), coRatedUserOfB2.append(b2[user])) for user in coRatedUser]
    averageOfB1 = sum(coRatedUserOfB1) / len(coRatedUserOfB1)
    averageOfB2 = sum(coRatedUserOfB2) / len(coRatedUserOfB2)

    a,b,c = 0,0,0
    for user in coRatedUser:
        a += (b1[user] - averageOfB1) * (b2[user]-averageOfB2)
        b += (b1[user] - averageOfB1) ** 2
        c += (b2[user] - averageOfB2) ** 2
    
    if a == 0 or b == 0 or c == 0:
        return 0 
    return a / (math.sqrt(b) * math.sqrt(c))

def main(argv):
    train_file = "train_review.json"
    model_file = "task3item.model"
    cf_type = "item_based"

    train_file = argv[0]
    model_file = argv[1]
    cf_type = argv[2]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("task3train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    data = sc.textFile(train_file).map(lambda x: json.loads(x)) \
            .map(lambda x: (x["user_id"], x["business_id"], x["stars"]))

    # Considering large datasets we may collect, we turn strings into indexes while processing
    # {idx: bussiness_id} -> len: 10253
    indexed_business = data.map(lambda x: x[1]).distinct().zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # {business_id: idx}
    indexed_business_inv = {v: k for k, v in indexed_business.items()}
    # {idx: user_id} -> len: 26184
    indexed_user = data.map(lambda x: x[0]).distinct().zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # {user_id: idx}
    indexed_user_inv = {v: k for k, v in indexed_user.items()}
    # turn data into all indexes
    data = data.map(lambda x: (indexed_user_inv[x[0]], indexed_business_inv[x[1]], x[2]))
    
    if cf_type == "item_based":
        # (business_idx, [(user_idx, rate)])
        matrix = data.map(lambda x: (x[1], (x[0], x[2]))).groupByKey() \
            .filter(lambda x: len(x[1]) >= CO_RATED_NUM)
        matrix_dict = matrix.mapValues(lambda x: dict(x)).collectAsMap()

        # (business_idx, [user_idx])
        # -> (business_idx, user_idx)
        # -> (user_idx, [business_idx])
        # -> (b1, b2)
        # -> filter & pearson correlation
        business_pairs = matrix.map(lambda x: (x[0], [t[0] for t in x[1]])).flatMap(lambda x: [(x[0], u) for u in x[1]]) \
            .map(lambda x: (x[1], x[0])).groupByKey() \
            .flatMap(lambda x: list(combinations(sorted(x[1]), 2))).distinct() \
            .filter(lambda x: x[0] < x[1] and filterByCoRatedNum(matrix_dict[x[0]], matrix_dict[x[1]])) \
            .map(lambda x: (x, pearson(matrix_dict[x[0]], matrix_dict[x[1]]))) \
            .filter(lambda x: x[1] > 0) \
            .collect()
        print(len(business_pairs))
        
        with open(model_file, 'w') as f:        
            for (k,v) in business_pairs:
                j = {"b1": indexed_business[k[0]], "b2": indexed_business[k[1]], "sim": v}
                f.write(json.dumps(j))
                f.write("\n")
            f.close()
        
    else: 
        ################################ MinHash & LSH ################################
        review = sc.textFile(train_file).map(lambda x: json.loads(x))
        # (user_id, [business_idx])
        characteristic_matrix = review.map(lambda x: (x["user_id"], indexed_business_inv[x["business_id"]])) \
                .groupByKey().mapValues(lambda x: sorted(x))

        #########  MinHash  #########
        m = len(indexed_business_inv)
        hash_params = hashFuncs(hashNum) # a,b parameter in each hash func
        signature_matrix = characteristic_matrix.map(lambda x: (x[0], hashIt(hashNum, m, x[1], hash_params)))

        #########  Locality Sensitive Hashing  #########
        candidates = signature_matrix.flatMap(lambda x: [(band, x[0]) for band in splitBands(x[1], bandNum)]) \
            .groupByKey() \
            .mapValues(lambda x: list(set(x))).filter(lambda x: len(x) > 1) \
            .flatMap(lambda x: list(combinations(x[1], 2))).map(lambda x: tuple(sorted(x))).distinct()

        #########  Jaccard Similarity  #########
        cMatrix = characteristic_matrix.mapValues(lambda x: list(x)).collectAsMap()
        # -> ((u1, u2), similarity)
        user_pairs = candidates.map(lambda x: (x, jaccard(x, cMatrix))) \
            .filter(lambda x: x[1] >= 0.01) \
            .filter(lambda x: filterByCoRatedNum1(cMatrix[x[0][0]], cMatrix[x[0][1]])) \
            .map(lambda x: (indexed_user_inv[x[0][0]], indexed_user_inv[x[0][1]]))
        # 2129 5646

        matrix_dict = data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey() \
                .filter(lambda x: len(x[1]) >= CO_RATED_NUM) \
                .mapValues(lambda x: dict(x)) \
                .collectAsMap()
        s = user_pairs.map(lambda x: (x, pearson(matrix_dict[x[0]], matrix_dict[x[1]]))).filter(lambda x: x[1] > 0).collect()
        # print(s.collect())
        
        with open(model_file, 'w') as f:
            for (k,v) in s:
                j = {"u1": indexed_user[k[0]], "u2": indexed_user[k[1]], "sim": v}
                f.write(json.dumps(j))
                f.write("\n")
            f.close()
        

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))