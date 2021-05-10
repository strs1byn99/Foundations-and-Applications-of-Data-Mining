from pyspark import SparkContext
from pyspark.mllib.fpm import FPGrowth
import sys
from collections import Counter, OrderedDict
from operator import add
import time, json

def generateBaskets(data, k):
    header = data.first()
    data = data.filter(lambda x: x != header)
    data = data.map(lambda x: (x.split(",")[0].strip(), x.split(",")[1].strip()))
    baskets = data.groupByKey().mapValues(set).mapValues(sorted)
    qualified = baskets.filter(lambda x: len(x[1]) > k).map(lambda x: x[1])
    return qualified

def main(argv):
    start = time.time()
    filter_threshold = int(argv[0])
    support = int(argv[1])
    in_file = argv[2]
    out_file = argv[3]

    # in_file = "user-business.csv"
    # filter_threshold = 70
    # support = 50
    # out_file = "myout3"

    sc = SparkContext(master="local[*]", appName="task3")
    data = sc.textFile(in_file)
    # generate baskets
    baskets = generateBaskets(data, filter_threshold)

    # run FP-Growth
    size = baskets.count()
    numPartitions = baskets.getNumPartitions()
    # print(size)
    # print(numPartitions)
    # if (size > 60): numPartitions = 60
    minSupport = float(support) / size
    model = FPGrowth.train(baskets, minSupport=minSupport, numPartitions=numPartitions)
    result = model.freqItemsets().collect()
    # print(len(result))

    # get task3 result
    l = [tuple(sorted(items)) for items,freq in result]
    l = set(l)  # set of tuples
    # print(len(l))

    duration = time.time() - start

    # get task2 result
    with open("task2_res_unduplicated.txt", "r") as fp:
        lines = fp.readlines()
        task2Num = lines[0].strip()
        l2 = set()
        jsonfile = json.loads(lines[1])
        l2 = [tuple(x) for x in jsonfile]
        l2 = set(l2)
    # print(task2Num) 
    # print(l2)

    intersection = len(l.intersection(l2))
    # print(intersection)

    start = time.time()
    with open(out_file, "w") as f:
        f.write("Task2,")
        f.write(str(task2Num))
        f.write("\nTask3,")
        f.write(str(len(result)))
        f.write("\nIntersection,")
        f.write(str(intersection))

    print("Duration: %f." % (duration + time.time() - start))
    
if __name__ == "__main__":
    main(sys.argv[1:])