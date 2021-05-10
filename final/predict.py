from surprise import dump
import time, json, sys
from utils import UID, IID, MODEL, RATING

start = time.time()
test_file = "test_review.json"
out_file = "out"
test_file = sys.argv[1]
out_file = sys.argv[2]

algo = dump.load(MODEL)[1]
with open(test_file, "r") as fin, open(out_file, "w") as fout:
    for line in fin.readlines():
        j = json.loads(line)
        x = (j[UID], j[IID])
        pred = algo.predict(x[0], x[1])[3]
        d = {UID: x[0], IID: x[1], RATING: pred}
        fout.write(json.dumps(d))
        fout.write("\n")
    fin.close()
    fout.close()
print("Duration: %f." % (time.time() - start))