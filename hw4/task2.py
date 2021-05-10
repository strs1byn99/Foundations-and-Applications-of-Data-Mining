from pyspark import SparkConf, SparkContext
import sys, time, json, random, math, copy
from collections import Counter, OrderedDict, defaultdict, deque
from itertools import combinations
from operator import add
import operator

def filteredByCoratedNum(u1_business, u2_business, t):
    return True if len(set(u1_business).intersection(set(u2_business))) >= t else False

def Girvan_Newman(root, adjacency, vertices):
    visited, queue = set(), deque([root])
    visited.add(root)
    
    parents_dict = defaultdict(set)
    level_dict, edges, path_count = dict(), dict(), dict()
    level_list = list()
    level = 0
    while queue:
        level += 1
        size = len(queue)
        level_sublist = []
        for i in range(size):
            # within each level, 1 node per iteration
            node = queue.popleft()
            level_dict[node] = level
            neighbours = adjacency[node]
            level_sublist.append(node)
            for neighbour in neighbours:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
                if level_dict.get(neighbour) == level - 1:
                    parents_dict[node].add(neighbour)
            # up to here: all parents are identified for this node
            path_count[node] = 1 if node == root else sum([path_count[p] for p in parents_dict[node]])
        level_list.append(level_sublist)
    
    credits = {each: 1 for each in vertices}
    level_list.reverse()
    for each_level in level_list[:-1]:
        for node in each_level:
            for parent in parents_dict[node]:
                fraction = path_count[parent] / path_count[node]
                value = credits[node] * fraction
                edges[tuple(sorted([node, parent]))] = value
                credits[parent] += value
    # print(edges)
    return list(edges.items())
    
def calculateModularity(communities, A, m):
    modularity = 0
    for community in communities:
        for i in community:
            for j in community:
                a = 1 if j in A[i] or i in A[j] else 0
                modularity += (a - (len(A[i]) * len(A[j]) / (2 * m)))
    return modularity / (2 * m)

def clustering(vertices, adjacency_dict):
    communities, visited = [], set()
    for vertex in vertices:
        if vertex not in visited:
            sub_visited = set([vertex])
            queue = deque([vertex])
            while queue:
                node = queue.popleft()
                for neighbour in adjacency_dict[node]:
                    if neighbour not in sub_visited:
                        sub_visited.add(neighbour)
                        queue.append(neighbour)
            visited = visited.union(sub_visited)
            communities.append(sub_visited)
    return communities

def main(argv):
    threshold = 7
    in_file = "ub_sample_data.csv"
    btwness_file = "myout1"
    cluster_file = "myout2"

    threshold = int(argv[0])
    in_file = argv[1]
    btwness_file = argv[2]
    cluster_file = argv[3]

    conf = SparkConf().setMaster("local[3]") \
        .setAppName("task3train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

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

    # adjacency list - BFS
    adjacency_dict = defaultdict(set)
    for (u1, u2) in pairs:
        adjacency_dict[u1].add(u2)
        adjacency_dict[u2].add(u1)
    # vertices - len: 222
    vertices = list(adjacency_dict.keys())

    ####################### Girvan Newman #######################
    btwness = sc.parallelize(vertices).map(lambda x: Girvan_Newman(x, adjacency_dict, vertices)) \
        .flatMap(lambda x: x).reduceByKey(add).map(lambda x: (x[0], x[1]/2)) \
        .sortBy(lambda x: (-x[1], x[0])) 

    btwness_out = btwness.map(lambda x: (tuple(sorted((indexed_user[x[0][0]], indexed_user[x[0][1]]))), x[1])) \
        .collect()
    
    with open(btwness_file, 'w') as f:
        for each in btwness_out:
            f.write(str(each)[1:-1] + '\n')
        f.close()

    #######################  Clustering  #######################

    # adjacency_matrix & degree_matrix
    A = copy.deepcopy(adjacency_dict)
    M = btwness.count()
    # edge count: 498
    max_modularity = -1
    count = M

    while True: 
        
        # remove the highest
        max_btwness = btwness.first()[1]
        toBeRemoved = btwness.filter(lambda x: x[1] == max_btwness).collect()
    
        # [((1366, 1670), 4234.0)]  
        for (edge, value) in toBeRemoved:
            adjacency_dict[edge[0]].remove(edge[1])
            adjacency_dict[edge[1]].remove(edge[0])
            count -= 1

        communities = clustering(vertices, adjacency_dict)
        # print(communities)
        
        modularity = calculateModularity(communities, A, M)
        # print(modularity)

        if modularity > max_modularity:
            max_modularity = modularity
            final_communities = communities

        if count == 0:
            break

        # redo Girvan Newman
        btwness = sc.parallelize(vertices).map(lambda x: Girvan_Newman(x, adjacency_dict, vertices)) \
            .flatMap(lambda x: x).reduceByKey(add).map(lambda x: (x[0], x[1]/2)) \
            .sortBy(lambda x: (-x[1], x[0]))

    # print(final_communities)
    # print(max_modularity)
    final = sc.parallelize(final_communities).map(lambda x: sorted([indexed_user[u] for u in x])) \
        .sortBy(lambda x: (len(x), x)).collect()
            
    with open(cluster_file, 'w') as fout:
        for each in final:
            fout.write(str(each)[1:-1] + '\n')
        fout.close()
    
if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))