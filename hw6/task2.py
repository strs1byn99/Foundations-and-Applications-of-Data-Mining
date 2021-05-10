from pyspark import SparkConf, SparkContext
import sys, time, json, random, math, copy, os, datetime, statistics
import csv
from pyspark.streaming.context import StreamingContext
from binascii import hexlify

WINDOW_LENGTH = 30
SLIDING_INTERVAL = 10
BATCH_DURATION = 5 

NUM_OF_BIT = 32
HASH_NUM = 200
P = 2147483647 # for hash func
K = 10

def hashFuncs():
    hashParams = []
    for i in range(HASH_NUM):
        a = random.randint(1, sys.maxsize)
        b = random.randint(1, sys.maxsize)
        hashParams.append((a,b))
    return hashParams

def Flajolet_Martin(rdd, hashParams, out_file):
    time = str(datetime.datetime.now())[:19]
    cities = ground_truth = rdd.distinct().collect()

    results = []
    for (a,b) in hashParams:
        max_zero_length = -math.inf
        for each in cities:
            idx = int(hexlify(each.encode("utf8")), 16)
            hashed = (a * idx + b) % P
            hashed = bin(hashed).zfill(NUM_OF_BIT)
            num_of_zeros = 0 if hashed == 0 else len(str(hashed)) - len(str(hashed).rstrip("0"))
            max_zero_length = max(num_of_zeros, max_zero_length)
        results.append(2 ** max_zero_length)
    # print(results)
    results = sorted(results)

    estimate = int(round((results[int(HASH_NUM/2-K-1)] + results[int(HASH_NUM/2+K-1)])/2, 0))

    print([time, len(ground_truth), estimate])
    with open(out_file, 'a') as fout: 
        output = csv.writer(fout)
        output.writerow([time, len(ground_truth), estimate])
        fout.close()

    return

def main(argv):
    port = 9999
    out_file = "myout2"

    port = int(argv[0])
    out_file = argv[1]

    with open(out_file, "w") as fout:
        fout.close()

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("Flajolet-Martin") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")

    ssc = StreamingContext(sc , BATCH_DURATION)
    stream = ssc.socketTextStream("localhost", port) \
        .window(WINDOW_LENGTH, SLIDING_INTERVAL) \
        .map(lambda x: json.loads(x))

    hashParams = hashFuncs()

    with open(out_file, 'a') as fout: 
        output = csv.writer(fout)
        output.writerow(["Time", "Ground Truth", "Estimation"])
        fout.close()

    stream.map(lambda x: x["city"]).filter(lambda x: x != "") \
        .foreachRDD(lambda rdd: Flajolet_Martin(rdd, hashParams, out_file))

    ssc.start()
    ssc.awaitTermination()

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))