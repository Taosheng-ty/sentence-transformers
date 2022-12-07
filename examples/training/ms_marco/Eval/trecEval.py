import os, time
from sentence_transformers import  util
import tarfile
if not os.path.exists("./utils/trec_eval-9.0.7/"):
    tar_filepath = "./utils/trec_eval-9.0.7.tar.gz"
    if not os.path.exists(tar_filepath):
        util.http_get('https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz', tar_filepath)
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path="./utils/")
    time.sleep(1)
    os.system("cd utils/trec_eval-9.0.7/ &&make")
    time.sleep(2)
jedgement=""
outfile=""
Eval="./utils/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10,20,50 "+jedgement+" "+outfile