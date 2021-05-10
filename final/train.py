from surprise import SVD, Dataset, Reader, dump
import pandas as pd
import time, json
from utils import UID, IID, MODEL, RATING

start = time.time()
file = "train_review.json"
file = "../resource/asnlib/publicdata/train_review.json"

# surprise documentation: 
# https://surprise.readthedocs.io/en/stable/getting_started.html
# RMSE below 1.22 - better than other algorithm (e.g. kNN, Item-based, User-based)
df = pd.read_json(open(file, "r", encoding="utf8"), lines=True)
df = df[[UID, IID, RATING]]
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5))
data = Dataset.load_from_df(df, reader)
trainset = data.build_full_trainset()
algo = SVD(n_epochs=25, lr_all=0.005, reg_all=0.1)
algo.fit(trainset)
dump.dump(MODEL, algo=algo)
print("Duration: %f." % (time.time() - start))