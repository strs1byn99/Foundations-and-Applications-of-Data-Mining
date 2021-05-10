from pyspark import SparkConf, SparkContext
import sys, time, json, random, math
from collections import Counter, OrderedDict, defaultdict, deque
from itertools import combinations
from operator import add
from graphframes import *
import os
from pyspark.sql import SQLContext

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

def filteredByCoratedNum(u1_business, u2_business, t):
    return True if len(set(u1_business).intersection(set(u2_business))) >= t else False

def main(argv):
    # spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py
    threshold = 7
    in_file = "ub_sample_data.csv"
    cluster_file = "task1_res"

    threshold = int(argv[0])
    in_file = argv[1]
    cluster_file = argv[2]

    conf = SparkConf().setMaster("local[3]") \
        .setAppName("task3train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sqlc = SQLContext(sc)

    data = sc.textFile(in_file)
    header = data.first()
    # (user, business)
    data = data.filter(lambda x: x != header) \
        .map(lambda x: (x.split(",")[0].strip(), x.split(",")[1].strip())) \
        .persist()

    # Considering large datasets we may collect, we turn strings into indexes while processing
    # {bussiness_id: idx} -> len: 9947
    indexed_business_inv = data.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    # {idx: user_id} -> len: 3374
    indexed_user = data.map(lambda x: x[0]).distinct().zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # {user_id: idx}
    indexed_user_inv = {v: k for k, v in indexed_user.items()}
    # turn data into all indexes
    data = data.map(lambda x: (indexed_user_inv[x[0]], indexed_business_inv[x[1]]))
    # data = data.map(lambda x: (x[0], x[1]))

    # (user, [businesss]) filtered by len([business]) >= 7
    # -> flatMap (business, [user])
    # -> combinations (u1, u2) pairs
    data = data.groupByKey().mapValues(lambda x: list(set(x))) \
        .filter(lambda x: len(x[1]) >= threshold) \
        .persist()
    # {user: [business]}
    matrix_dict = data.collectAsMap()
    
    pairs = data.flatMap(lambda x: [(b, x[0]) for b in x[1]]).groupByKey() \
        .flatMap(lambda x: list(combinations(sorted(set(x[1])), 2))).distinct() \
        .filter(lambda x: filteredByCoratedNum(matrix_dict[x[0]], matrix_dict[x[1]], threshold)) \
        .collect()
    data.unpersist()
    # len: 498

    vertices = set()
    for (u1, u2) in pairs:
        vertices.add(tuple([u2]))
        vertices.add(tuple([u1]))
    # vertices - len: 222
    vertices = list(vertices)

    ######################### LPA #########################
    vertices_df = sqlc.createDataFrame(vertices, ['id'])
    edges_df = sqlc.createDataFrame(pairs, ['src', 'dst'])
    graph = GraphFrame(vertices_df, edges_df)
    labels = graph.labelPropagation(maxIter=5)
    # Sample: [Row(id=505, label=1)]
    # returns a DataFrame with nodes and a label denoting which community that node belongs in
    communities = labels.rdd.map(lambda x: (x[1], x[0])).groupByKey() \
        .map(lambda x: sorted([indexed_user[u] for u in x[1]])) \
        .sortBy(lambda x: (len(x), x)).collect()

    with open(cluster_file, 'w') as f:
        for each in communities:
            f.write(str(each)[1:-1] + '\n')
        f.close()

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))