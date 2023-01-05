"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models,CrossEncoder
import logging
import sys
import os
import tarfile
import argparse
from utils.data import loadMSCorpus,loadDevMSqueries,loadDevRelMSQrels
import itertools
from collections import defaultdict
import pytrec_eval
from tqdm import tqdm
from utils.ranking import BM25FirstPhase
import numpy as np
import argparse
import torch
import random
import json
import os, time
from sentence_transformers import  util
import tarfile
import csv
from str2bool import str2bool
from utils.data import *
from utils.eval import *
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
data_folder =os.path.join(os.path.expanduser('~'), '.cache/')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='distilbert-base-uncased')
    parser.add_argument("--log_dir",default="output/log", help="where to store the model")
    parser.add_argument("--gpu", type=int, default=None, nargs="+", help="used gpu")
    parser.add_argument("--rerun", type=str2bool, default=False, help="reurn to generate rnklist or not, default no")
    parser.add_argument("--msdev", type=str2bool, default=True,  help="evaluate on ms dev or not, default yes.")
    parser.add_argument("--evalFunc", type=str, default="DualEncoderRanklist",  help="evaluate function")
    parser.add_argument("--data", type=str, default="0+1+2", help="data set to eval")
    parser.add_argument("--modelClass", type=str, default="SentenceTransformer", help="data set to eval")
    args = parser.parse_args()
    dataEval=args.data.split("+")
    dataEval=[int(i) for i in dataEval]
    # if args.gpu is None:
    # devices=list(range(torch.cuda.device_count()))
    #     gpu=str(random.choice(devices))
    #     print(gpu,devices)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # print(devices,"devices",flush=True)
    print(args)
    # The  model we want to fine-tune
    model_name = args.model_name
    # model_name="msmarco-distilbert-base-tas-b"
    # model_name="../output/mse-huggingfaceHard10EpochDist/171600"
    # model_name="../output/mseRetrainRound2/740000"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # if 
    # model=CrossEncoder(model_name)
    # else:
    model = globals()[args.modelClass](model_name)
    # model.model.to(model._target_device)
    # model.eval()
    data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
    # data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCOToy')
    
    # data_folder ="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/msmarco-data"
    os.makedirs(data_folder,exist_ok=True)
    AggResults= defaultdict(list)
    batch_size=32
    if args.msdev:
        # ir_evaluator=LoadMSDevEvaluator(data_folder)
        dev_queries=loadDevMSqueries(data_folder)
        dev_rel_docs=loadDevRelMSQrels(data_folder,dev_queries)
        jedgement = os.path.join(data_folder, 'qrels.dev.tsv')
        # Read passages
        corpus=loadMSCorpus(data_folder)
        # corpus= {k: corpus[k] for k in list(corpus.keys())[:10000]}
        dataLogFolder=os.path.join(args.log_dir,"MSMARCORetrieval")
        os.makedirs(dataLogFolder,exist_ok=True)
        # ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs)
        ir_evaluator = evaluatorSbert2(dev_queries, corpus, dev_rel_docs,show_progress_bar=True,batch_size=64)
        retrieResult=ir_evaluator.compute_metrices(model,cutoff=100,trecOutputPath=dataLogFolder) 
        AggResults["MSMARCOPassDot"].append(retrieResult["dot_score"]["ndcg@k"])
        AggResults["MSMARCOPassCos"].append(retrieResult["cos_sim"]["ndcg@k"])
        # hitsOrigId, _=getSbertRanklist(model,dev_queries,corpus,outputPath=dataLogFolder,reEmb=True,batch_size=batch_size)
        
        Eval="./utils/trec_eval-9.0.7/trec_eval  -M 10 -m  recip_rank "+jedgement+" "+dataLogFolder+"/dot_score-trec.rnk"
        print(Eval)
        os.system(Eval)
        # retrieResult=ir_evaluator.compute_metrices(model)
        # print(ir_evaluator(model))
        print(retrieResult)
        # retrieResult=ir_evaluator.compute_metrices(model)
        # print(retrieResult,"model metrices")
        # AggResults["MSMARCOPassDot"].append(retrieResult[]["mrr@k"])

    AggResults["iterations"].append(0)
    dataNames=list(GlobalDataset.keys())
    dataNames=[dataNames[i] for i in dataEval]
    # dataNames=list(GlobalDataset.keys())

    # dataNames=["scifact"]
    for dataName in  dataNames:
        print(f"processing data {dataName}",flush=True)
        evaluator,relevant_docs=qrels2Evaluator(dataName,{'ndcg_cut.10','ndcg_cut.100',"map"})
        if "trec" in GlobalDataset[dataName] and GlobalDataset[dataName]:
            dev_queries=loadDevMSqueries(data_folder)
            corpus=loadMSCorpus(data_folder)
            candidateSet,queries,Corpus=loadEvalRanklistTrec(dataName,dev_queries,corpus)
        else:
            candidateSet,queries,Corpus=loadEvalRanklist(dataName)
        queriedFiltered=list(queries.keys()&relevant_docs.keys())
        queries={queryEach:queries[queryEach] for queryEach in queriedFiltered}
        dataLogFolder=os.path.join(args.log_dir,dataName)
        trecOutfile=os.path.join(dataLogFolder,"output.trec.csv")
        msOutfile=os.path.join(dataLogFolder,"output.ms.csv")
        if not os.path.isfile(trecOutfile) or args.rerun:
            run=globals()[args.evalFunc](model,candidateSet,queries,Corpus,batch_size=batch_size)
        else:
            with open(trecOutfile, 'r') as f_run:
                run = pytrec_eval.parse_run(f_run)            
        EvalResults=evaluator.evaluate(run)
        RealCalMetrics=list(EvalResults.values())[0].keys()
        for measure in sorted(RealCalMetrics):
            AggResults[dataName+measure].append(pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                        for query_measures in EvalResults.values()]))

        os.makedirs(dataLogFolder, exist_ok=True)
        runsDict2trec(run,trecOutfile)
        runsDict2Msmarco(run,msOutfile)
        jedgement=GlobalDataset[dataName]["testQrels"]
        Eval="./utils/trec_eval-9.0.7/trec_eval  -M 10 -m  recip_rank "+jedgement+" "+trecOutfile
        print(Eval)
        os.system(Eval)
        Eval="./utils/trec_eval-9.0.7/trec_eval  -M 100 -m  recip_rank "+jedgement+" "+trecOutfile
        print(Eval)
        os.system(Eval)
        Eval="python  ./utils/msmarco_passage_eval.py  "+jedgement+" "+msOutfile
        print(Eval)
        os.system(Eval)
        Eval="python  ./utils/evalSbert.py  "+jedgement+" "+trecOutfile
        print(Eval)
        os.system(Eval)
    print(AggResults)
    with open(args.log_dir+"/AggResults.jjson", "w") as outfile:
        # outfile.write(ending)
        json.dump(AggResults,outfile)
    
    