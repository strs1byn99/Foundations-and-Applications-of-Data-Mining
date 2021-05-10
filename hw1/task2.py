import sys
from pyspark import SparkContext
import json

def evalWithSpark(review, business, n):
    # (business_id, (stars, count))
    review_stars = review.map(lambda x: (x['business_id'], (x['stars'], 1))) \
        .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))
    # (business_id, [categories])
    business_category = business.filter(lambda x: x['categories'] is not None) \
        .map(lambda x: (x['business_id'], [c.strip() for c in (x['categories'].split(","))]))
    """
    join -> (business_id, ((stars, count), [categories]))
    map -> ([categories], (stars, count))
    flatMap -> (category, (stars, count))
    then calculate average
    then sort
    """
    return review_stars.join(business_category) \
        .map(lambda x: (x[1][1], x[1][0])) \
        .flatMap(lambda x: [(c, x[1]) for c in x[0]]) \
        .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])) \
        .mapValues(lambda x: float(format(float(x[0] / x[1]), '.1f'))) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(int(n))

def evalWithoutSpark(review_file, business_file, n):
    with open(review_file, 'r') as review_f:
        review_json = [json.loads(review) for review in review_f]

    # review_dict: {business_id: [stars]}
    review_dict = dict()
    for each in review_json:
        business_id = each['business_id']
        stars = each['stars']
        if business_id not in review_dict:
            review_dict[business_id] = list()
        review_dict[business_id].append(stars)

    with open(business_file, 'r') as business_f:
        business_json = [json.loads(business) for business in business_f]

    # business_dict: {business_id: [categories]}
    business_dict = dict()
    for each in business_json:
        business_id = each['business_id']
        categories_raw = each['categories']
        categories = list()
        if categories_raw is not None:
            categories = [c.strip() for c in (categories_raw.split(","))]
        if business_id not in business_dict:
            business_dict[business_id] = list()
        for c in categories:
            business_dict[business_id].append(c)

    # joined by business_id: [([categories], [stars])]
    joined = list(map(lambda x: (business_dict.get(x[0]), (x[1])), review_dict.items()))

    # category_dict: {category: [stars]}
    category_dict = dict()
    for (categories, stars) in joined:
        if categories is not None:
            for category in categories:
                if category not in category_dict:
                    category_dict[category] = list()
                for s in stars:
                    category_dict[category].append(s)

    # calculate average -> (category, avg_stars)
    average = list(map(lambda x: (x[0], float(format(float(sum(x[1]) / len(x[1])), '.1f'))), category_dict.items()))
    sorted_avg = sorted(average, key=lambda x: (-x[1],x[0]))
    return sorted_avg[0:int(n)]

def main(argv):
    review_file = argv[0]
    business_file = argv[1]
    out_file = argv[2]
    if_spark = argv[3]
    n = argv[4]
    
    result = dict()
    if (if_spark == "no_spark"):
        result['result'] = evalWithoutSpark(review_file, business_file, n)
    else:
        sc = SparkContext("local[*]", "task2")
        review = sc.textFile(review_file).map(lambda x: json.loads(x))
        business = sc.textFile(business_file).map(lambda x: json.loads(x))
        result['result'] = evalWithSpark(review, business, n)

    with open(out_file, 'w') as f_out:
        f_out.write(json.dumps(result))
    f_out.close()

if __name__ == '__main__':
    main(sys.argv[1:])