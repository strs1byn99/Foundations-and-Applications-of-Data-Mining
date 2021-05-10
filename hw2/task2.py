from pyspark import SparkContext
import sys
from collections import Counter, OrderedDict, defaultdict
from operator import add
import time, json

def generateBaskets(data, k):
    header = data.first()
    data = data.filter(lambda x: x != header)
    data = data.map(lambda x: (x.split(",")[0].strip(), x.split(",")[1].strip()))
    baskets = data.groupByKey().mapValues(set).mapValues(sorted)
    qualified = baskets.filter(lambda x: len(x[1]) > k)
    # for each in qualified.collect():
    #     print(each)
    return qualified

def moreThanApriori(partition, support, size):
    pbaskets = list(partition)
    psupport = float(support) * len(pbaskets) / size
    
    # singletons
    frequent = []
    for b in pbaskets:
        frequent.extend(b[1])
    C = Counter(frequent)
    singletons = set(sorted([c for c in C if C[c] >= psupport]))
    frequent = [tuple([x]) for x in singletons]
    
    prevFrequent = frequent
    while len(prevFrequent) > 0:
        levelTupleCount = defaultdict(lambda: 0) # { tuple : count }
        for basket in pbaskets:
            basket = set(basket[1]).intersection(singletons)  # filter by singletons
            alreadyCounted = set()  # { tuple1, tuple2 ... } already-counted tuples
            for c in prevFrequent:
                c = set(c)
                if c.issubset(basket) and len(c) < len(basket):    
                    toBeAdded = basket - c       # set of to-be-added elem
                    for each in toBeAdded:       # add 1 elem to get a new candidate
                        n = c.union(set([each])) # new candidate as set
                        n = tuple(sorted(n))     # sorted candidate as tuple
                        if n in alreadyCounted: continue 
                        else: alreadyCounted.add(n) # check existence of candidate in basket
                        levelTupleCount[n] += 1  # increment counter
        # filter by local threshold
        prevFrequent = [c for c in levelTupleCount if levelTupleCount[c] >= psupport]
        # add new candidates
        frequent.extend(prevFrequent)        

    return frequent

def countFrequent(basket, candidates):
    l = [tuple(c) for c in candidates if set(c).issubset(basket)]
    return [(c, 1) for c in l]

def formatCandidates(candidates):
    result = OrderedDict()
    for each in candidates:
        key = len(each)
        if key not in result:
            result[key] = ""
        result[key] += str(each).replace("[", "(").replace("]", ")") + ","
    
    result = [result[x][:-1] for x in result]
    return result

def main(argv):
    filter_threshold = int(argv[0])
    support = int(argv[1])
    in_file = argv[2]
    out_file = argv[3]

    # in_file = "user-business.csv"
    # filter_threshold = 70
    # support = 50
    # out_file = "myout2"

    sc = SparkContext(master="local[*]", appName="task2")
    data = sc.textFile(in_file)
    # generate baskets
    baskets = generateBaskets(data, filter_threshold)
    
    # Apriori on partitions
    size = baskets.count()
    # print(baskets.getNumPartitions())

    # Phase 1
    t_rdd = baskets.mapPartitions(lambda x: moreThanApriori(x, support, size)).distinct() \
        .map(lambda x: list(x))
    t = t_rdd.collect()
    t = sorted(t)

    # Phase 2
    s_rdd = baskets.flatMap(lambda x: countFrequent(x[1], t))
    s = s_rdd.reduceByKey(add).filter(lambda x: x[1] >= support) \
        .map(lambda x: sorted(x[0])).collect()
    s = sorted(s)
    # print(len(s))

    # print(t)
    # print("*****************")
    # print(s)
    
    with open(out_file, "w") as f:
        f.write("Candidates:\n")
        res1 = formatCandidates(t)
        for line in res1:
            f.write(line + "\n\n")
        f.write("Frequent Itemsets:\n")
        res2 = formatCandidates(s)
        f.write(res2[0])
        for line in res2[1:]:
            f.write("\n\n" + line)
    return s
        
if __name__ == "__main__":
    start = time.time()
    s = main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))

    """
    write results to task2_res_unduplicated.txt
    serve for task3
    """ 
    with open("task2_res_unduplicated.txt", "w") as fp:
        fp.write(str(len(s)))
        fp.write("\n")
        fp.write(json.dumps(s))