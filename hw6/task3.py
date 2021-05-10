from pyspark import SparkConf, SparkContext
import sys, time, json, random, math, csv, tweepy
from collections import defaultdict

API_KEY = ""
API_SECRET_KEY = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
TRACK = ["China", "United States", "Covid", "Musk", "weather", "news",
    "bitcoin", "Nasdaq", "stock", "money", "crypto"]

class MyStreamListener(tweepy.StreamListener):

    def __init__(self, out_file):
        tweepy.StreamListener.__init__(self)
        self.tweet_count = 0
        self.tag_dict = defaultdict(lambda: 0)
        self.tweet_sample_list = [] # [[tag1, tag2], [...]]
        
        self.out_file = out_file
        with open(self.out_file, "w") as fout:
            fout.close()
        
    def on_status(self, status):
        tag_list = status.entities.get("hashtags")
        if tag_list != None and len(tag_list) > 0:
            self.tweet_count += 1
            tags = self.___extractTag___(tag_list)
            if self.tweet_count < 100:
                self.___addTag___(tags)
            else:
                if self.___ifReplace___():
                    self.___updateTag___(tags)
                self.___write___()
    
    def ___extractTag___(self, taglist):
        return [tagdict["text"] for tagdict in taglist]

    def ___addTag___(self, taglist):
        for tag in taglist:
            self.tag_dict[tag] += 1
        self.tweet_sample_list.append(taglist)

    def ___updateTag___(self, taglist):
        pos = random.randint(0,99)
        toBeRemoved = self.tweet_sample_list[pos]
        # decrement 
        for each in toBeRemoved:
            self.tag_dict[each] -= 1
            if self.tag_dict[each] == 0: self.tag_dict.pop(each, -1)
        # update in sample list
        self.tweet_sample_list[pos] = taglist
        # increment 
        for tag in taglist:
            self.tag_dict[tag] += 1
        
    def ___ifReplace___(self):
        return random.random() < 100 / self.tweet_count

    def ___write___(self):
        sorted_dict = sorted(self.tag_dict.items(), key=lambda kv: (-kv[1], kv[0]))
        top3 = sorted(set(self.tag_dict.values()), reverse=True)[:3]
        with open(self.out_file, "a") as fout:
            fout.write("The number of tweets with tags from the beginning: {}\n".format(self.tweet_count))
            for k,v in sorted_dict:
                if v in top3:
                    fout.write(k + " : " + str(v) + "\n")
            fout.write("\n")
            fout.close()
        # print("count: " + str(self.tweet_count) + " top3: " + str(top3))

def main(argv):

    out_file = "myout3"
    out_file = argv[1]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("twitterStreaming") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    myStreamListener = MyStreamListener(out_file=out_file)
    myStream = tweepy.Stream(auth=auth, listener=myStreamListener)
    myStream.filter(track=TRACK, languages=["en"])

if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    print("Duration: %f." % (time.time() - start))