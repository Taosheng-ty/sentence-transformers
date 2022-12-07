from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import sys
import json
import os
from collections import defaultdict
# scriptPath=os.path.dirname(os.path.abspath(__file__))
# os.chdir(scriptPath+"/../")
from eval import *
trecRun= "/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/training_ms-marco_cross-encoder-v2-microsoft-MiniLM-L12-H384-uncased-2022-12-03_13-25-05/985000/Eval/MSDev/output.trec.csv"

# class sbertEval()

def fromTrecRuntoSbertRun(trecRun):
    sbertRun=defaultdict(list)
    for line in open(trecRun):
            query_id, _, doc_id, rank, score, _ = line.strip().split(' ')
            sbertRun[query_id].append({'corpus_id': doc_id, 'score': score})
    return sbertRun

sbertRun=fromTrecRuntoSbertRun(trecRun)
data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
ir_evaluator=LoadMSDevEvaluator(data_folder)
ir_evaluator.compute_metrics(sbertRun)
    
    