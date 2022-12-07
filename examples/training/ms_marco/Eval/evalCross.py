"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from eval import *    
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
    # docEmb=model.encode(list(Corpus.values()),batch_size=batch_size,show_progress_bar=True)  # encode sentence
    # queryEmb=model.encode(list(queries.values()),batch_size=batch_size,show_progress_bar=True)
    # queries = dict(zip(list(queries.keys()),queryEmb))
    # docs=dict(zip(list(Corpus.keys()),docEmb))
    # run={}
    # for qid in candidateSet:
    #     run[qid]={}
    #     for pid in candidateSet[qid]:
    #         score=np.sum(docs[pid]*queries[qid])
    #         run[qid][pid]=float(score)
    return run
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='distilbert-base-uncased')
    parser.add_argument("--log_dir",default="output/log", help="where to store the model")
    parser.add_argument("--gpu", type=int, default=None, nargs="+", help="used gpu")
    parser.add_argument("--msdev", type=str2bool, default=True,  help="evaluate on ms dev or not, default yes.")
    parser.add_argument("--evalFunc", type=str, default=True,  help="evaluate on ms dev or not, default yes.")
    args = parser.parse_args()
    # if args.gpu is None:
    devices=list(range(torch.cuda.device_count()))
    #     gpu=str(random.choice(devices))
    #     print(gpu,devices)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(devices,"devices")
    print(args)
    # The  model we want to fine-tune
    model_name = args.model_name
    # model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    # model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    # model_name="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/training_ms-marco_cross-encoder-v2-microsoft-MiniLM-L12-H384-uncased-2022-12-02_14-58-49/"

    # model_name="output/mse-huggingfaceHard10EpochDist/171600"
    # model_name="../output/log/0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    model = CrossEncoder(model_name,max_length=512)
    # model.to(model._target_device)
    model.model.eval()
    # data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
    # # data_folder ="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/msmarco-data"
    # os.makedirs(data_folder,exist_ok=True)
    AggResults= defaultdict(list)
    AggResults["iterations"].append(0)
    dataNames=list(GlobalDataset.keys())[:3]
    # dataNames=list(GlobalDataset.keys())
    batch_size=128
    # dataNames=["scifact"]
    for dataName in  dataNames:
        evaluator,relevant_docs=qrels2Evaluator(dataName,metrics=pytrec_eval.supported_measures)
        candidateSet,queries,Corpus=loadEvalRanklist(dataName)
        queriedFiltered=list(queries.keys()&relevant_docs.keys())
        queries={queryEach:queries[queryEach] for queryEach in queriedFiltered}
        if not os.path.isfile(trecOutfile) or args.rerun:
            run=crossEncoderRanklist(model,candidateSet,queries,Corpus,batch_size=batch_size)
        else:
            with open(trecOutfile, 'r') as f_run:
                run = pytrec_eval.parse_run(f_run)         
        
        EvalResults=evaluator.evaluate(run)
        # print(EvalResults)
        RealCalMetrics=list(EvalResults.values())[0].keys()
        for measure in sorted(RealCalMetrics):
            AggResults[dataName+measure].append(pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                        for query_measures in EvalResults.values()]))
        # print(AggResults)
        dataLogFolder=os.path.join(args.log_dir,dataName)
        trecOutfile=os.path.join(dataLogFolder,"output.trec.csv")
        msOutfile=os.path.join(dataLogFolder,"output.ms.csv")
        os.makedirs(dataLogFolder, exist_ok=True)
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
        
        # outfile=os.path.join(args.log_dir,dataName,"output.tsv")
        # runsDict2trec(run,outfile)
        # jedgement=GlobalDataset[dataName]["testQrels"]
        # Eval="./utils/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10,20,50 "+jedgement+" "+outfile
        # os.system(Eval)
    print(AggResults)
    with open(args.log_dir+"/AggResults.jjson", "w") as outfile:
        # outfile.write(ending)
        json.dump(AggResults,outfile)  
    
    