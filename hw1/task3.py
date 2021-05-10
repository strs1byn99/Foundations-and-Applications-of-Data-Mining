import sys
from pyspark import SparkContext
import json

def evalByPartitions(review, n, partition_num=-1):
    result = dict()
    rdd = review.map(lambda x: (x['business_id'], 1))
    if partition_num != -1:
        rdd = rdd.partitionBy(partition_num, lambda x: ord(x[0]))
    result['n_partitions'] = rdd.getNumPartitions()
    result['n_items'] = rdd.glom().map(lambda x: len(x)).collect()
    result['result'] = rdd.reduceByKey(lambda x,y: x+y) \
        .filter(lambda x: x[1] > int(n)) \
        .collect()
    return result

def main(argv):
    review_file = argv[0]
    out_file = argv[1]
    partition_type = argv[2]
    partition_num = argv[3]
    n = argv[4]

    sc = SparkContext("local[*]", "task3")
    review = sc.textFile(review_file).map(lambda x: json.loads(x))
    
    result = dict()
    if partition_type == "customized":
        result = evalByPartitions(review, n, partition_num=int(partition_num))
    else:
        result = evalByPartitions(review, n)

    with open(out_file, 'w') as f_out:
        f_out.write(json.dumps(result))
    f_out.close()

if __name__ == '__main__':
    main(sys.argv[1:])