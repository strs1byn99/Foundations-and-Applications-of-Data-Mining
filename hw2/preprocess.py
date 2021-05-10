from pyspark import SparkContext
import csv
import json

def main():
    # business_file = "./data/business.json"
    # review_file = "./data/review.json"# 
    business_file = "./business.json"
    review_file = "./review.json"
    # business_file = "./new.json"
    # review_file = "./new1.json"
    
    sc = SparkContext(master="local[*]", appName="preprocess")
    review = sc.textFile(review_file).map(lambda x: json.loads(x))
    business = sc.textFile(business_file).map(lambda x: json.loads(x))

    business_filtered = business.filter(lambda x: x['state'] == "NV") \
                    .map(lambda x: (x['business_id'], x['state'])).distinct()
    review_filtered = review.map(lambda x: (x['business_id'], x['user_id']))
    joined = review_filtered.join(business_filtered)
    result = joined.map(lambda x: (x[1][0], x[0]))
    t = result.collect()
    # print(t)

    with open('user-business.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(["user_id", "business_id"])
        write.writerows(t)

if __name__ == '__main__':
    main()