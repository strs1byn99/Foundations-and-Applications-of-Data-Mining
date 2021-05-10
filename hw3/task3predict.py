from pyspark import SparkConf, SparkContext
import sys, time, json, random, math
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from operator import add
import time, os

NEIGHBOUR_NUM = 10

def predict(target_idx, model_dict, business_rated_by_user, indexed_dict, rate_avg):
    neighbors = [] # [(rate, weight)]
    for (neighbor_idx, rate) in business_rated_by_user:
        business_pair = tuple(sorted([target_idx, neighbor_idx]))
        if model_dict.get(business_pair):
            neighbors.append((rate, model_dict.get(business_pair)))
    selectedNeighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:NEIGHBOUR_NUM]
    a = sum(map(lambda x: x[0] * x[1], selectedNeighbors))
    b = sum(map(lambda x: abs(x[1]), selectedNeighbors))
    if a == 0 or b == 0:
        return rate_avg[indexed_dict.get(target_idx, "UNK")]
    return a / b

def predict1(target_idx, model_dict, rated_user_on_business, indexed_dict, rate_avg):
    target_id = indexed_dict.get(target_idx, "UNK")
    neighbors = [] # [(rate, average_rate, weight)]
    for (neighbor_idx, rate) in rated_user_on_business:
        user_pair = tuple(sorted([target_idx, neighbor_idx]))
        average_rate = rate_avg[indexed_dict.get(neighbor_idx, "UNK")]
        if model_dict.get(user_pair):
            neighbors.append((rate, average_rate, model_dict.get(user_pair)))
    neighbors = sorted(neighbors, key=lambda x: x[2], reverse=True)[:NEIGHBOUR_NUM]
    a = sum(map(lambda x: (x[0] - x[1]) * x[2], neighbors))
    b = sum(map(lambda x: abs(x[2]), neighbors))
    if a == 0 or b == 0:
        return (rate_avg[target_id])
    return rate_avg[target_id] + (a / b)

def main(argv):
    train_file = "train_review.json"
    test_file = "test_review.json"
    model_file = "task3item.model"
    out_file = "task3item.predict"
    cf_type = "item_based"
    business_avg_file = "business_avg.json"
    user_avg_file = "user_avg.json"

    train_file = argv[0]
    test_file = argv[1]
    model_file = argv[2]
    out_file = argv[3]
    cf_type = argv[4]
    business_avg_file = "../resource/asnlib/publicdata/business_avg.json"
    user_avg_file = "../resource/asnlib/publicdata/user_avg.json"

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("task3predict") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    # (user_id, business_id, stars)
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
    # data = data.map(lambda x: (x[0], x[1], x[2]))
    
    if (cf_type == "item_based"):
        business_avg = sc.textFile(business_avg_file).map(lambda x: json.loads(x)).flatMap(lambda x: x.items()) \
            .collectAsMap()

        # ((b1_idx, b2_idx), weight)) - get weights and neighbours by business_idx
        model = sc.textFile(model_file).map(lambda x: json.loads(x)).map(lambda x: (x["b1"], x["b2"], x["sim"]))
        model_dict = model.map(lambda x: ((indexed_business_inv[x[0]], indexed_business_inv[x[1]]), x[2])) \
                .collectAsMap()

        # (user_idx, [(business_idx, stars)]) - get user's rate on different businesses
        user_business = data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey() \
            .mapValues(lambda x: list(set(x)))
        
        # (user_idx, business_idx)
        test = sc.textFile(test_file).map(lambda x: json.loads(x)) \
            .map(lambda x: (indexed_user_inv.get(x["user_id"]), indexed_business_inv.get(x["business_id"]))) \
            .filter(lambda x: x[0] != None and x[1] != None)

        # predict - according to each user_business pair
        # (user_idx, (b1_idx, [(b2_idx, rate)]))
        res = test.leftOuterJoin(user_business).map(lambda x: ((x[0], x[1][0]), predict(x[1][0], model_dict, x[1][1], indexed_business, business_avg))) \
            .collect()

    else:
        user_avg = sc.textFile(user_avg_file).map(lambda x: json.loads(x)).flatMap(lambda x: x.items()) \
            .collectAsMap()

        # ((u1_idx, u2_idx), weight)) - get weights and neighbours by business_idx
        model = sc.textFile(model_file).map(lambda x: json.loads(x)).map(lambda x: (x["u1"], x["u2"], x["sim"]))
        model_dict = model.map(lambda x: ((indexed_user_inv[x[0]], indexed_user_inv[x[1]]), x[2])) \
                .collectAsMap()

        # (business_idx, [(user_idx, stars)]) - get users' rate on each business
        user_business = data.map(lambda x: (x[1], (x[0], x[2]))).groupByKey() \
            .mapValues(lambda x: list(set(x)))

        # (business_idx, user_idx)
        test = sc.textFile(test_file).map(lambda x: json.loads(x)) \
            .map(lambda x: (indexed_business_inv.get(x["business_id"]), indexed_user_inv.get(x["user_id"]))) \
            .filter(lambda x: x[0] != None and x[1] != None)

        # predict - according to each business_user pair
        # (business_idx, (u1_idx, [(u2_idx, rate)]))
        res = test.leftOuterJoin(user_business).map(lambda x: ((x[1][0], x[0]), predict1(x[1][0], model_dict, x[1][1], indexed_user, user_avg))) \
            .collect()

    with open(out_file, 'w') as f:
        for (k,v) in res:
            j = {"user_id": indexed_user.get(k[0], k[0]), "business_id": indexed_business.get(k[1], k[1]), "stars": v}
            f.write(json.dumps(j))
            f.write("\n")
        f.close()

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))