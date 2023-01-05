import sys
import json
import os
from torch import Tensor
import torch
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm, trange
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../")
from collections import defaultdict

# scriptPath=os.path.dirname(os.path.abspath(__file__))

# from eval import *
from typing import List, Tuple, Dict, Set, Callable
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import numpy as np
class evaluatorSbert(evaluation.InformationRetrievalEvaluator):
    def __init__(self,*args, **kwargs):
        super().__init__(corpus={1:"None"},queries={1:"None"},*args, **kwargs)
        self.numQ=len(list(self.relevant_docs.keys()))
    def compute_metrics(self, queries_result_dict):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}
        NumEvalQ=len(list(queries_result_dict.keys()))
        print("total num of queries is {numQ}, and we evaluate {NumEvalQ} queries".format(numQ=self.numQ,NumEvalQ=NumEvalQ))
        # Compute scores on results
        counter=0
        for query_itr in list(queries_result_dict.keys()):
            counter+=1
            query_id = query_itr

            # Sort scores
            top_hits = sorted(queries_result_dict[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        # if rank!=0:
                        #     print(query_itr,hit['corpus_id'],top_hits[0:k_val],query_relevant_docs)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /=NumEvalQ

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= NumEvalQ

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])
        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}


# class sbertEval()

def fromTrecRuntoSbertRun(trecRun):
    sbertRun=defaultdict(list)
    for line in open(trecRun):
            query_id, _, doc_id, rank, score, _ = line.strip().split(' ')
            sbertRun[query_id].append({'corpus_id': doc_id, 'score': float(score)})
    return sbertRun

def qrelsLoad(qrels_filepath):
    relevant_docs = defaultdict(dict)
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, score = line.strip().split()
            score = float(score)
            if score > 0:
                relevant_docs[qid][pid] = score
    return relevant_docs


def main():
    """Command line:
    # python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """

    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
    # if len(sys.argv) > 0:
    # #     path_to_reference = "/home/collab/u1368791/.cache/MSMARCO/qrels.dev.tsv"
    # #     path_to_candidate = "/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/training_ms-marco_cross-encoder-v2-microsoft-MiniLM-L12-H384-uncased-2022-12-03_13-25-05/985000/Eval/MSDev/output.ms.csv"
    # #     path_to_candidate = "/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-margin_mse--home-collab-u1368791-largefiles-TaoFiles-sentence-transformers-examples-training-ms_marco-output-mrr0.36Biencoder-235740--batch_size_64-2022-12-03_23-48-15/350000/Eval/MSDev/output.trec.csv"
    #     path_to_reference = "/home/collab/u1368791/.cache/MSMARCO_DL19Pass/2019qrels-pass.txt" 
    #     path_to_candidate ="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-margin_mse--home-collab-u1368791-largefiles-TaoFiles-sentence-transformers-examples-training-ms_marco-output-mrr0.36Biencoder-235740--batch_size_64-2022-12-03_23-48-15/650000/Eval/DL19/output.trec.csv"
        sbertRun=fromTrecRuntoSbertRun(path_to_candidate)
        dev_rel_docs=qrelsLoad(path_to_reference)

        ir_evaluator = evaluatorSbert( relevant_docs=dev_rel_docs,
                                                                show_progress_bar=True,
                                                                precision_recall_at_k=[10, 100],
                                                                name="msmarco dev")
        results=ir_evaluator.compute_metrics(sbertRun)
        print(results)

    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking>')
        exit()
    
if __name__ == '__main__':
    main()

