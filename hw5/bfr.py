from pyspark import SparkConf, SparkContext
import sys, time, json, random, math, copy, os
from collections import Counter, OrderedDict, defaultdict, deque
from itertools import combinations
from operator import add
import csv

MULTIPLE = 3 # for a larger K
ALPHA = 2 # for Mahalanobis Distance
SAMPLE = 0.3

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

def euclidean_distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(p1, p2)]))

def locate_centroid(cluster, dp):
    points = [dp[each] for each in cluster]
    c = [sum(d) / len(d) for d in zip(*points)]
    return c

# dp: {key: [values]}
def KMeans(K, dp):
    # pick k random points
    random_k = random.sample(list(dp), K)
    centroids, clusters = dict(), dict()
    centroids_ori_idx = dict()
    for idx,key in enumerate(random_k):
        centroids[idx] = dp[key]
        centroids_ori_idx[idx] = key
    # print(centroids)
    
    i = 0
    while True:
        # current clusters
        clusters = {idx:set() for (idx, point) in centroids.items()}
        # add original centroid/k points
        if (i == 0):
            for (idx, key) in centroids_ori_idx.items():
                clusters[idx].add(key)
        for (idx, point) in dp.items():
            # calculate euclidean distance from each point to center
            dists = [euclidean_distance(point, c) for c in centroids.values()]
            # add point to the nearest cluster
            closest_idx = dists.index(min(dists))
            clusters[closest_idx].add(idx)

        # update centroids
        new_centroids = dict()
        for (idx, cluster) in clusters.items():
            c = locate_centroid(cluster, dp)
            new_centroids[idx] = c
        # print(new_centroids)

        # check if centroid positions change
        shouldBreak = True
        for (idx, point) in centroids.items():
            # compare each point/centroids - total K points/centroids
            oldc = set(map(lambda x: round(x), point))
            newc = set(map(lambda x: round(x), new_centroids[idx]))
            if (len(oldc - newc) != 0):
                shouldBreak = False
                break
        if i == 100: shouldBreak = True

        i += 1
        if shouldBreak: break
        # print("iteration: " + str(i))
        centroids = new_centroids
    
    return clusters, centroids

def summarize(num_of_dim, dp, cluster):
    n = len(cluster)
    SUM = [0] * num_of_dim
    SUMSQ = [0] * num_of_dim
    std = [0] * num_of_dim

    points = [dp[each] for each in cluster]
    SUM = [sum(d) for d in zip(*points)]
    SUMSQ = [sum(val**2 for val in d) for d in zip(*points)]
    # (sumsq, sum)
    std = [math.sqrt(ssq/n - (s/n)**2) for (ssq, s) in zip(SUMSQ, SUM)]
    return (SUM, SUMSQ, std, n)

def mahalanobis_distance(centroid, point, std):
    return math.sqrt(sum([((a - b) / s) ** 2 if s != 0 else (a-b)**2 for (a,b,s) in zip(centroid, point, std) ]))

def find_nearest_cluster(point, centroids, summary, threshold):
    # calculate mahalanobis dist from each to all DS centroids & find the nearest DS
    min_dist = math.inf
    assigned = -1
    for (idx, c) in centroids.items():
        std = summary[idx][2]
        dist = mahalanobis_distance(c, point, std)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            assigned = idx
    return assigned

def assign_to_nearest_clusters(dp, DS, DS_centroids, DS_summary, CS, CS_centroids, CS_summary, RS, threshold):
    for (point_idx, point) in dp.items():
        # find the nearest cluster, -1 if not found
        assigned = find_nearest_cluster(point, DS_centroids, DS_summary, threshold)
        # can be put into a DS
        if assigned > -1:
            updated_n = DS_summary[assigned][3] + 1
            updated_SUM = [a + b for (a,b) in zip(DS_summary[assigned][0], point)]
            updated_SUMSQ = [a + b**2 for (a,b) in zip(DS_summary[assigned][1], point)]
            updated_centroid = [x / updated_n for x in updated_SUM]
            updated_std = [math.sqrt(ssq/updated_n - (s/updated_n)**2) for (ssq, s) in zip(updated_SUMSQ, updated_SUM)]
            # update DS_centroids & DS_summary & DS dict
            DS_centroids[assigned] = updated_centroid
            DS_summary[assigned] = [updated_SUM, updated_SUMSQ, updated_std, updated_n]
            DS[point_idx] = assigned
        else:
            # cannot be put into any DS, check if okay for any CS, -1 if not found
            assigned = find_nearest_cluster(point, CS_centroids, CS_summary, threshold)
            if assigned > -1:
                # update CS_centroids & CS_summary & CS dict
                updated_n = CS_summary[assigned][3] + 1
                updated_SUM = [a + b for (a,b) in zip(CS_summary[assigned][0], point)]
                updated_SUMSQ = [a + b**2 for (a,b) in zip(CS_summary[assigned][1], point)]
                updated_centroid = [x / updated_n for x in updated_SUM]
                updated_std = [math.sqrt(ssq/updated_n - (s/updated_n)**2) for (ssq, s) in zip(updated_SUMSQ, updated_SUM)]
                CS_centroids[assigned] = updated_centroid
                CS_summary[assigned] = [updated_SUM, updated_SUMSQ, updated_std, updated_n]
                CS[point_idx] = assigned
            else:
                # cannot be put into any CS, add to RS
                RS[point_idx] = point

def save_intermediate_records(ds, cs, rs, iteration, ds_sum, cs_sum):   
    return [iteration, len(ds_sum), len(ds), len(cs_sum), len(cs), len(rs)]

def merge_CSs(centroids, summary, CS, threshold):
    centroid_idxs = list(centroids)
    pairs = combinations(centroid_idxs, 2)
    discarded = set()
    for (idx1, idx2) in pairs:
        if idx1 not in discarded and idx2 not in discarded:
            c1 = centroids[idx1]
            c2 = centroids[idx2]
            std = summary[idx1][2]
            dist = mahalanobis_distance(c1, c2, std)
            if dist < threshold:
                # merge idx2 into idx1
                updated_n = summary[idx1][3] + summary[idx2][3]
                SUM1 = summary[idx1][0]
                SUM2 = summary[idx2][0]
                updated_SUM = [a+b for (a,b) in zip(SUM1, SUM2)]
                SUMSQ1 = summary[idx1][1]
                SUMSQ2 = summary[idx2][1]
                updated_SUMSQ = [a+b for (a,b) in zip(SUMSQ1, SUMSQ2)]
                updated_centroid = [x / updated_n for x in updated_SUM]
                updated_std = [math.sqrt(ssq/updated_n - (s/updated_n)**2) for (ssq, s) in zip(updated_SUMSQ, updated_SUM)]
                # update summary, centroids, sets
                centroids[idx1] = updated_centroid
                summary[idx1] = [updated_SUM, updated_SUMSQ, updated_std, updated_n]
                # move points from idx2 to idx1
                CS_copy = copy.deepcopy(CS)
                for (k,v) in CS_copy.items():
                    if (v == idx2):
                        CS[k] = idx1
                # discard idx2
                discarded.add(idx2)
                summary.pop(idx2, -1)
                centroids.pop(idx2, -1)

def merge_CS_to_DS(CS_centroids, CS_summary, DS_centroids, DS_summary, DS, reversed_CS, CS, RS):
    # for each CS, calculate its dist to each DS & do merge
    for (cs_idx, cs_centroid) in CS_centroids.items():
        assigned = find_nearest_cluster(cs_centroid, DS_centroids, DS_summary, math.inf)
        # merge CS into DS
        updated_n = DS_summary[assigned][3] + CS_summary[cs_idx][3]
        SUM1 = DS_summary[assigned][0]
        SUM2 = CS_summary[cs_idx][0]
        updated_SUM = [a+b for (a,b) in zip(SUM1, SUM2)]
        SUMSQ1 = DS_summary[assigned][1]
        SUMSQ2 = CS_summary[cs_idx][1]
        updated_SUMSQ = [a+b for (a,b) in zip(SUMSQ1, SUMSQ2)]
        updated_centroid = [x / updated_n for x in updated_SUM]
        updated_std = [math.sqrt(ssq/updated_n - (s/updated_n)**2) for (ssq, s) in zip(updated_SUMSQ, updated_SUM)]
        # update DS_centroids, DS_summary, DS
        DS_centroids[assigned] = updated_centroid
        DS_summary[assigned] = [updated_SUM, updated_SUMSQ, updated_std, updated_n]
        CS_summary.pop(cs_idx, -1)
        for each in reversed_CS[cs_idx]:
            DS[each] = assigned
            CS.pop(each, -1)

    # for each RS, calculate its dist to each DS & do merge
    RS_copy = copy.deepcopy(RS)
    for (point_idx, point) in RS_copy.items():
        assigned = find_nearest_cluster(point, DS_centroids, DS_summary, math.inf)
        # merge RS into DS
        updated_n = DS_summary[assigned][3] + 1
        updated_SUM = [a + b for (a,b) in zip(DS_summary[assigned][0], point)]
        updated_SUMSQ = [a + b**2 for (a,b) in zip(DS_summary[assigned][1], point)]
        updated_centroid = [x / updated_n for x in updated_SUM]
        updated_std = [math.sqrt(ssq/updated_n - (s/updated_n)**2) for (ssq, s) in zip(updated_SUMSQ, updated_SUM)]
        # update DS_centroids, DS_summary, DS
        DS_centroids[assigned] = updated_centroid
        DS_summary[assigned] = [updated_SUM, updated_SUMSQ, updated_std, updated_n]
        DS[point_idx] = assigned
        RS.pop(point_idx, -1)

def main(argv):
    input_dir_path = "./test2"
    K = int("10")
    out_cluster_file = "myout1"
    out_intermediate_file = "myout1_"

    input_dir_path = argv[0]
    K = int(argv[1])
    out_cluster_file = argv[2]
    out_intermediate_file = argv[3]

    intermediate_results = list()

    files = sorted(os.listdir(input_dir_path))
    file_paths = [input_dir_path + "/" + x for x in files]
    
    conf = SparkConf().setMaster("local[*]") \
        .setAppName("bfr") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    DS_summary, DS, DS_centroids = dict(), dict(), dict()
    CS_summary, CS, CS_centroids = dict(), dict(), dict()
    RS = dict()

    ################################### first-load data points ###################################
    data = sc.textFile(file_paths[0]) \
        .map(lambda x: (int(x.split(",")[0]), [float(d) for d in x.split(",")[1:]])) \
        .collectAsMap()
    num_of_dim = len(list(data.values())[0])
    threshold = ALPHA * math.sqrt(num_of_dim)
    data_keys = list(data)
    num_of_point = len(data)
    
    # sample for initial clusters
    random.seed(20)
    sample = random.sample(data_keys, int(num_of_point * SAMPLE))
    non_sample = set(data_keys)- set(sample)
    sample_dict = {each: data[each] for each in sample}
    non_sample_dict = {each: data[each] for each in non_sample}

    # k-means on sample - {0: {1,324,523...}, 1: {...}}
    # summarize initial DS
    initial_DS, DS_centroids = KMeans(K, sample_dict)
    for (idx, cluster) in initial_DS.items():
        (SUM, SUMSQ, std, lenOfCluster) = summarize(num_of_dim, sample_dict, cluster)
        DS_summary[idx] = [SUM, SUMSQ, std, lenOfCluster]
        for each in cluster:
            DS[each] = idx

    assign_to_nearest_clusters(data, DS, DS_centroids, DS_summary,
                                    CS, CS_centroids, CS_summary, RS, threshold)

    # generate CS by running KMeans with larger K
    # summarize CS, determine RS
    initial_CS, CS_centroids = KMeans(min(MULTIPLE * K, len(RS)), RS)
    for (idx, cluster) in initial_CS.items():
        if len(cluster) > 1:
            (SUM, SUMSQ, std, lenOfCluster) = summarize(num_of_dim, RS, cluster)
            CS_summary[idx] = [SUM, SUMSQ, std, lenOfCluster]
            for each in cluster:
                CS[each] = idx
        else:
            if len(cluster) == 1:
                rs_idx = list(cluster)[0]
                RS[rs_idx] = data[rs_idx]
            CS_centroids.pop(idx, -1)
    # print(CS_summary)

    r = save_intermediate_records(DS, CS, RS, 1, DS_summary, CS_summary)
    intermediate_results.append(r)
    print(r)

    ################################### second+ loads data points ###################################
    cluster_num = MULTIPLE * K
    it = 2
    for f in file_paths[1:]:
        data = sc.textFile(f) \
            .map(lambda x: (int(x.split(",")[0]), [float(d) for d in x.split(",")[1:]])) \
            .collectAsMap()

        # new points assigned to DS, CS using mahanalobis
        assign_to_nearest_clusters(data, DS, DS_centroids, DS_summary,
                                        CS, CS_centroids, CS_summary, RS, threshold)

        # run larger K-means on RS
        initial_CS, new_CS_centroids = KMeans(min(MULTIPLE * K, len(RS)), RS)
        for (idx, cluster) in initial_CS.items():
            if len(cluster) > 1:
                # add new CS & move some RS to CS
                (SUM, SUMSQ, std, lenOfCluster) = summarize(num_of_dim, RS, cluster)
                CS_summary[cluster_num] = [SUM, SUMSQ, std, lenOfCluster]
                for each in cluster:
                    if each in RS:
                        RS.pop(each, -1)
                    CS[each] = cluster_num
                CS_centroids[cluster_num] = new_CS_centroids[idx]
                cluster_num += 1
            else:
                # remaining RS & new RS
                if len(cluster) == 1:
                    rs_idx = list(cluster)[0]
                    if (rs_idx not in RS and rs_idx in data):
                        RS[rs_idx] = data[rs_idx]
        # print(CS_summary)
        # print(list(CS_centroids))

        # merge CSs
        merge_CSs(CS_centroids, CS_summary, CS, threshold)

        if it == len(os.listdir(input_dir_path)):
            reversed_CS = {}
            for k,v in CS.items():
                if v not in reversed_CS: reversed_CS[v] = [k]
                else: reversed_CS[v].append(k)
            merge_CS_to_DS(CS_centroids, CS_summary, DS_centroids, DS_summary, DS, reversed_CS, CS, RS)

        r = save_intermediate_records(DS, CS, RS, it, DS_summary, CS_summary)
        intermediate_results.append(r)
        print(r)
        it += 1

    with open(out_intermediate_file,"w") as fout:
        s = ["round_id","nof_cluster_discard","nof_point_discard","nof_cluster_compression","nof_point_compression","nof_point_retained"]
        writer = csv.writer(fout)
        writer.writerow(s)
        for i in intermediate_results:
            writer.writerow(i)
        fout.close()

    final = dict(sorted(DS.items(), key=lambda item: (item[1], item[0])))
    with open(out_cluster_file, 'w') as fout:
        json.dump(final, fout)
        fout.close()

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))