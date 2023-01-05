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
from typing import List, Tuple, Dict, Set, Callable
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import numpy as np
from tqdm import tqdm, trange
from torch import Tensor
import torch
import logging
import psutil
logger = logging.getLogger(__name__)
scriptPath=os.path.dirname(os.path.abspath(__file__))
# os.chdir(scriptPath)
trecEvalPath=os.path.join(scriptPath,"trec_eval-9.0.7/")
trecEvalTarPath=os.path.join(scriptPath,"trec_eval-9.0.7.tar.gz")
if not os.path.exists(trecEvalPath):
    tar_filepath = trecEvalTarPath
    if not os.path.exists(tar_filepath):
        util.http_get('https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz', tar_filepath)
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path="./")
    time.sleep(1)
    os.system(f"cd {trecEvalPath}/ &&make")
    time.sleep(2)  
data_folder =os.path.join(os.path.expanduser('~'), '.cache/')
GlobalDataset={"DL19":{"testQuery-Passage":data_folder+"MSMARCO_DL19Pass/msmarco-passagetest2019-top1000.tsv", 
                "testQrels":data_folder+"MSMARCO_DL19Pass/2019qrels-pass.txt"},
        "DL20":{"testQuery-Passage":data_folder+"MSMARCO_DL20Pass/msmarco-passagetest2020-top1000.tsv",
                "testQrels":data_folder+"MSMARCO_DL20Pass/2020qrels-pass.txt"},
        "MSDev":{"testQuery-Passage":data_folder+"MSMARCO/top1000.dev.tsv",
                "testQrels":data_folder+"MSMARCO/qrels.dev.tsv"},
        "disbert0.37":{"testQuery-Passage":"/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/mseRetrainRound2/740000/Eval/MSMARCORetrieval/dot_score-trec.rnk",
                "testQrels":data_folder+"MSMARCO/qrels.dev.tsv","trec":True},
        "disbert0.355":{"testQuery-Passage":"/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/biencoder-vali/epoch-33steps--1-Score0.5761924603174603/Eval/MSMARCORetrieval/dot_score-trec.rnk",
                "testQrels":data_folder+"MSMARCO/qrels.dev.tsv","trec":True},
        "bert0.381":{"testQuery-Passage":"/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/Eval/output/msmarco-bert-base-dot-v5/Eval/MSMARCORetrieval/dot_score-trec.rnk",
                "testQrels":data_folder+"MSMARCO/qrels.dev.tsv",
                "trec":True}
        }
BeirDataStorePath=data_folder+"BeirDatasets/"
datasetsName = ["scifact","nfcorpus","fiqa","arguana","scidocs"]
# datasetsName = ["scifact"]
BM25ResultsPath=BeirDataStorePath+"BM25InitialRnk/"
# for datasetCurName in datasetsName:
#     url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(datasetCurName)
    
#     DataDir = os.path.join(BeirDataStorePath, datasetCurName)
#     if not os.path.exists(DataDir):
#         data_path = util.download_and_unzip(url, BeirDataStorePath)
#     GlobalDataset[datasetCurName]={}
#     GlobalDataset[datasetCurName]["isBeir"]=True
#     GlobalDataset[datasetCurName]["dataPath"]=DataDir
#     DatasetBM25Path=os.path.join(BM25ResultsPath,datasetCurName)
# #     TestQuery_Passage=os.path.join(DatasetBM25Path,"BM25Top1000.tsv")
# #     DevQuery_Passage=os.path.join(DatasetBM25Path,"BM25Top1000.tsv")
#     splits=["test"]
#     for split in splits:
#         splitPath=os.path.join(DatasetBM25Path,split)
#         os.makedirs(splitPath, exist_ok=True)
#         BM25Path=os.path.join(splitPath,"top1000.tsv")
#         QresPath=os.path.join(splitPath,"qrels.tsv")
#         if not os.path.exists(BM25Path) or not os.path.exists(QresPath): 
#             BM25FirstPhase(DataDir,BM25Path,QresPath,split=split)
#         GlobalDataset[datasetCurName][split+"Query-Passage"]=BM25Path
#         GlobalDataset[datasetCurName][split+"Qrels"]=QresPath


# importing libraries
import os
import psutil

# inner psutil function
def process_memory():
	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()
	return mem_info.rss


def LoadMSDevEvaluator(data_folder,Sizelimit=None,*args, **kwargs):
    ### Data files
    
    os.makedirs(data_folder, exist_ok=True)
    ### Load data

    corpus = {}             #Our corpus pid => passage
    dev_queries = {}        #Our dev queries. qid => query
    dev_rel_docs = {}       #Mapping qid => set with relevant pids

    dev_queries=loadDevMSqueries(data_folder)
    dev_rel_docs=loadDevRelMSQrels(data_folder,dev_queries)
    # Read passages
    corpus=loadMSCorpus(data_folder)
    
    if Sizelimit is not None:
        neededPid=list(itertools.chain.from_iterable(dev_rel_docs.values()))
        corpusNew={pid: corpus[pid] for pid in neededPid}
        sizeCur=0
        for pid in corpus:
            if pid in corpusNew:
                continue
            corpusNew[pid]=corpus[pid]
            sizeCur=len(corpusNew.keys())
            if sizeCur>Sizelimit:
                corpus=corpusNew
                logging.info("Eval Corpus: {}".format(len(corpus)))
                break
            
    ## Run evaluator
    logging.info("Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(corpus)))

    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[10, 100],
                                                            name="msmarco dev",
                                                            *args, **kwargs)
    return ir_evaluator


def qrels2Evaluator(dataset,metrics= pytrec_eval.supported_measures):
    #Read which passages are relevant
    relevant_docs = defaultdict(lambda: defaultdict(int))
    # qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')

    # if not os.path.exists(qrels_filepath):
    #     logging.info("Download "+os.path.basename(qrels_fpath))
    #     util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)

    qrels_filepath=GlobalDataset[dataset]["testQrels"]
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, score = line.strip().split()
            score = int(score)
            if score > 0:
                relevant_docs[qid][pid] = score
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, metrics)
    return evaluator,relevant_docs
def loadEvalRanklist(dataset):
    queries_passage_filepath=GlobalDataset[dataset]["testQuery-Passage"]
    candidateSet= defaultdict(list)
    num_lines = sum(1 for line in open(queries_passage_filepath))
    n=1
    Corpus=defaultdict(str)
    queries=defaultdict(str)
    with open(queries_passage_filepath, 'r', encoding='utf8') as fIn:
        for line in tqdm(fIn, desc ="Loading data",total=num_lines):
            qid, pid, query,passage = line.strip().split("\t")
            candidateSet[qid].append(pid)
            queries[qid]=query
            Corpus[pid]=passage
    return candidateSet,queries,Corpus
def loadEvalRanklistTrec(dataset,queryOrig,CorpusOrig):
    queries_passage_filepath=GlobalDataset[dataset]["testQuery-Passage"]
    candidateSet= defaultdict(list)
    num_lines = sum(1 for line in open(queries_passage_filepath))
    n=1
    Corpus=defaultdict(str)
    queries=defaultdict(str)
    with open(queries_passage_filepath, 'r', encoding='utf8') as fIn:
        for line in tqdm(fIn, desc ="Loading data",total=num_lines):
            qid,_, pid,rank,score,_ = line.strip().split()
            candidateSet[qid].append(pid)
            queries[qid]=queryOrig[qid]
            Corpus[pid]=CorpusOrig[pid]
    return candidateSet,queries,Corpus
            #  {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}
#     return candidateSet
def DualEncoderRanklist(model,candidateSet,queries,Corpus,batch_size=32):
    docEmb=model.encode(list(Corpus.values()),batch_size=batch_size,show_progress_bar=True)  # encode sentence
    queryEmb=model.encode(list(queries.values()),batch_size=batch_size,show_progress_bar=True)
    queries = dict(zip(list(queries.keys()),queryEmb))
    docs=dict(zip(list(Corpus.keys()),docEmb))
    run={}
    for qid in queries:
        run[qid]={}
        for pid in candidateSet[qid]:
            score=np.sum(docs[pid]*queries[qid])
            run[qid][pid]=float(score)
    return run
# def convertSbertEval(run):
#     for query_itr, query in enumerate(run.keys()):
#         for sub_corpus_id, score in run[query].items():
#             corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
#             queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})
    
def runsDict2trec(run, outfile):
    with open(outfile, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter=' ')
        NumQuery=len(run.keys())
        for qid in tqdm(run, desc ="Generating ranklists",total=NumQuery):
            
            ranklist=list(run[qid].items())
            ranklist.sort(key=lambda x: -x[1])
            for rank, (Pid,score) in enumerate(ranklist):
                tsv_writer.writerow([qid,"Q0",Pid,rank+1,score,"List"])  ## output to trec format  
def runsDict2Msmarco(run, outfile):
    with open(outfile, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        NumQuery=len(run.keys())
        for qid in tqdm(run, desc ="Generating ranklists",total=NumQuery):
            
            ranklist=list(run[qid].items())
            ranklist.sort(key=lambda x: -x[1])
            for rank, (Pid,score) in enumerate(ranklist):
                tsv_writer.writerow([qid,Pid,rank+1])  ## output to trec format  
def crossEncoderRanklist(model,candidateSet,queries,Corpus,batch_size=32,*args, **kwargs):
    run={}
    NumQuery=len(queries.keys())
    for qid in tqdm(queries, desc ="Generating ranklists",total=NumQuery):
        query = queries[qid]

        cand = candidateSet[qid]
        pids = [c for c in cand]
        corpus_sentences = [Corpus[c] for c in cand]

        cross_inp = [[query, sent] for sent in corpus_sentences]

        if model.config.num_labels > 1: #Cross-Encoder that predict more than 1 score, we use the last and apply softmax
            cross_scores = model.predict(cross_inp, apply_softmax=True,show_progress_bar=False,batch_size=batch_size)[:, 1].tolist()
        else:
            cross_scores = model.predict(cross_inp,show_progress_bar=False,batch_size=batch_size).tolist()

        cross_scores_sparse = {}
        for idx, pid in enumerate(pids):
            cross_scores_sparse[pid] = cross_scores[idx]

        sparse_scores = cross_scores_sparse
        run[qid] = {}
        for pid in sparse_scores:
            run[qid][pid] = float(sparse_scores[pid])
    return run             
def getSbertRanklist(model,queries,corpus,outputPath=None,reEmb=False,batch_size=32,top_k=100):
    pids=list(corpus.keys())
    pidConvert={i:key for i, key in enumerate(pids)}
    qidConvert={i:key for i, key in enumerate(queries.keys())}

    corpusList= list(corpus.values())

    
    if outputPath is not None:
        os.makedirs(outputPath, exist_ok=True)
        embPath=os.path.join(outputPath,"emb.npy")
        if os.path.isfile(embPath) and not reEmb:
            emb=np.load(embPath)
            queryEmb=emb["q"]
            docEmb=emb["d"]
        else:
            docEmb=model.encode(corpusList,batch_size=batch_size,show_progress_bar=True)  # encode sentence
            queryEmb=model.encode(list(queries.values()),batch_size=batch_size,show_progress_bar=True)
            np.save(embPath,{"q":queryEmb,"d":docEmb})
            # torch.save({"q":queryEmb,"d":docEmb},embPath) 
    corpus_embeddings = docEmb
    # corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    query_embeddings = queryEmb
    # query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score,top_k=top_k)   
    hitsOrigId=[[{'corpus_id': pidConvert[pscore["corpus_id"]], 'score': pscore["score"]}   for pscore in q ]    for q in hits]
    hits2trec={qidConvert[qid]:{pidConvert[pscore["corpus_id"]]:pscore["score"]  for pscore in q }    for qid,q in enumerate(hits)}
    if outputPath is not None:
        runsDict2trec(hits2trec,outfile=os.path.join(outputPath,"trec.rnk"))
    return hitsOrigId, hits2trec


class evaluatorSbert2(evaluation.InformationRetrievalEvaluator):
    def  GenerateRnk(self, model, corpus_model = None, corpus_embeddings: Tensor = None,cutoff=200,trecOutputPath=None):
        if corpus_model is None:
            corpus_model = model

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))
        assert max_k <= cutoff, f"the number of ranklist cutoff is {cutoff}, which should be greater than evaluation needed cutoff {max_k}"
        max_k=cutoff     
        # Compute embedding for the queries
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        #Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000,flush=True)
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            #Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                #Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})
            #Sort and strip to top_k results
        for  score_functionname in queries_result_list:    
            queries_result_listEach=queries_result_list[score_functionname]
            for idx in range(len(queries_result_listEach)):
                queries_result_listEach[idx] = sorted(queries_result_listEach[idx], key=lambda x: x['score'], reverse=True)
                queries_result_listEach[idx] = queries_result_listEach[idx][0:max_k]
            if trecOutputPath is not None:
                hits2trec={self.queries_ids[qid]:{pscore["corpus_id"]:pscore["score"]  for pscore in q }    for qid,q in enumerate(queries_result_listEach)}
                runsDict2trec(hits2trec,outfile=os.path.join(trecOutputPath,score_functionname+"-trec.rnk"))   
                
        return  queries_result_list       
    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None,cutoff=200,trecOutputPath=None) -> Dict[str, float]:

        queries_result_list=self.GenerateRnk(model=model,corpus_model =corpus_model ,corpus_embeddings=corpus_embeddings,cutoff=cutoff,trecOutputPath=trecOutputPath)
        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        #Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        #Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])
        return scores