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
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
data_folder =os.path.join(os.path.expanduser('~'), '.cache/')
GlobalDataset={"DL19":{"testQuery-Passage":data_folder+"MSMARCO_DL19Pass/msmarco-passagetest2019-top1000.tsv", 
                "testQrels":data_folder+"MSMARCO_DL19Pass/2019qrels-pass.txt"},
        "DL20":{"testQuery-Passage":data_folder+"MSMARCO_DL20Pass/msmarco-passagetest2020-top1000.tsv",
                "testQrels":data_folder+"MSMARCO_DL20Pass/2020qrels-pass.txt"},
        "MSDev":{"testQuery-Passage":data_folder+"MSMARCO/top1000.dev.tsv",
                "testQrels":data_folder+"MSMARCO/qrels.dev.tsv"}
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
if not os.path.exists("./utils/trec_eval-9.0.7/"):
    tar_filepath = "./utils/trec_eval-9.0.7.tar.gz"
    if not os.path.exists(tar_filepath):
        util.http_get('https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz', tar_filepath)
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path="./utils/")
    time.sleep(1)
    os.system("cd utils/trec_eval-9.0.7/ &&make")
    time.sleep(2)  
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
def crossEncoderRanklist(model,candidateSet,queries,Corpus,*args, **kwargs):
    run={}
    for qid in queries:
        query = queries[qid]

        cand = candidateSet[qid]
        pids = [c for c in cand]
        corpus_sentences = [Corpus[c] for c in cand]

        cross_inp = [[query, sent] for sent in corpus_sentences]

        if model.config.num_labels > 1: #Cross-Encoder that predict more than 1 score, we use the last and apply softmax
            cross_scores = model.predict(cross_inp, apply_softmax=True)[:, 1].tolist()
        else:
            cross_scores = model.predict(cross_inp).tolist()

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
        embPath=os.path.join(outputPath,"emb.pt")
        if os.path.isfile(embPath) and not reEmb:
            emb=torch.load(embPath)
            queryEmb=emb["q"]
            docEmb=emb["d"]
        else:
            docEmb=model.encode(corpusList,batch_size=batch_size,show_progress_bar=True,convert_to_tensor=True)  # encode sentence
            queryEmb=model.encode(list(queries.values()),batch_size=batch_size,show_progress_bar=True,convert_to_tensor=True)
            torch.save({"q":queryEmb,"d":docEmb},embPath) 
    corpus_embeddings = docEmb.to('cuda')
    # corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    query_embeddings = queryEmb.to('cuda')
    # query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score,top_k=top_k)   
    hitsOrigId=[[{'corpus_id': pidConvert[pscore["corpus_id"]], 'score': pscore["score"]}   for pscore in q ]    for q in hits]
    hits2trec={qidConvert[qid]:{pidConvert[pscore["corpus_id"]]:pscore["score"]  for pscore in q }    for qid,q in enumerate(hits)}
    if outputPath is not None:
        runsDict2trec(hits2trec,outfile=os.path.join(outputPath,"trec.rnk"))
    return hitsOrigId, hits2trec
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='distilbert-base-uncased')
    parser.add_argument("--log_dir",default="output/log", help="where to store the model")
    parser.add_argument("--gpu", type=int, default=None, nargs="+", help="used gpu")
    parser.add_argument("--rerun", type=str2bool, default=False, help="reurn to generate rnklist or not, default no")
    parser.add_argument("--msdev", type=str2bool, default=True,  help="evaluate on ms dev or not, default yes.")
    parser.add_argument("--evalFunc", type=str, default="DualEncoderRanklist",  help="evaluate function")
    args = parser.parse_args()
    # if args.gpu is None:
    devices=list(range(torch.cuda.device_count()))
    #     gpu=str(random.choice(devices))
    #     print(gpu,devices)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(devices,"devices",flush=True)
    print(args)
    # The  model we want to fine-tune
    model_name = args.model_name
    # model_name="msmarco-distilbert-base-tas-b"
    # model_name="output/mse-huggingfaceHard10EpochDist/171600"
    # model_name="../output/log/0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = SentenceTransformer(model_name)
    model.to(model._target_device)
    model.eval()
    data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
    # data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCOToy')
    
    # data_folder ="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/msmarco-data"
    os.makedirs(data_folder,exist_ok=True)
    AggResults= defaultdict(list)
    batch_size=128
    if args.msdev:
        # ir_evaluator=LoadMSDevEvaluator(data_folder)
        dev_queries=loadDevMSqueries(data_folder)
        dev_rel_docs=loadDevRelMSQrels(data_folder,dev_queries)
        jedgement = os.path.join(data_folder, 'qrels.dev.tsv')
        # Read passages
        corpus=loadMSCorpus(data_folder)
        # corpus= {k: corpus[k] for k in list(corpus.keys())[:10000]}
        dataLogFolder=os.path.join(args.log_dir,"MSMARCORetrieval")
        hitsOrigId, _=getSbertRanklist(model,dev_queries,corpus,outputPath=dataLogFolder,reEmb=True,batch_size=batch_size)
        
        Eval="./utils/trec_eval-9.0.7/trec_eval  -M 10 -m  recip_rank "+jedgement+" "+dataLogFolder+"/trec.rnk"
        print(Eval)
        os.system(Eval)
        # retrieResult=ir_evaluator.compute_metrices(model)
        # print(ir_evaluator(model))
        
        ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs)
        retrieResult=ir_evaluator.compute_metrics(hitsOrigId)
        AggResults["MSMARCOPassDot"].append(retrieResult["mrr@k"])
        print(retrieResult)
        # retrieResult=ir_evaluator.compute_metrices(model)
        # print(retrieResult,"model metrices")
        # AggResults["MSMARCOPassDot"].append(retrieResult[]["mrr@k"])
    AggResults["iterations"].append(0)
    dataNames=list(GlobalDataset.keys())[:3]
    # dataNames=list(GlobalDataset.keys())

    # dataNames=["scifact"]
    for dataName in  dataNames:
        evaluator,relevant_docs=qrels2Evaluator(dataName,{'ndcg_cut.10','ndcg_cut.100',"map"})
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
    
    