import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../")
import BatchExpLaunch.tools as tools

import os
import json
import os
import time
rootpath="output/mse-huggingfaceHard10EpochDist/"
rootpath="output/mse-huggingfaceHard10EpochCo-con/"
rootpath="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-margin_mse-distilbert-base-uncased-batch_size_64-2022-11-30_11-40-37/"
rootpath="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-2022-11-30_11-40-45/"
rootpath="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-margin_mse--home-collab-u1368791-largefiles-TaoFiles-sentence-transformers-examples-training-ms_marco-output-train_bi-encoder-margin_mse-distilbert-base-uncased-batch_size_64-2022-11-30_11-40-37-235740-batch_size_64-2022-12-02_11-38-25/"
rootpath="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/mseRetrainRound2/"
# rootpath="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/biencoder-vali/"
temp="scripts/slurmtemp.slurm"
# subdir=["50000","100000","120000","140000",]
subdir=["210000","220000","230000","235740"]
subdir=["350000","370000","390000"]
subdir=["740000"]

# subdir=["124800"]
# subdir=["200000","220000","230000","235740","210000"]
# subdir=os.listdir(path=rootpath)
for path in subdir:
    curFolder=os.path.join(rootpath,path)
    if os.path.isdir(curFolder):
        # cmd="slurmRun --cmd='python eval.py  --model_name={rootpath}/{path} --log_dir={rootpath}/{path}/Eval' --template={temp} --outputDir={rootpath}/{path}/Eval".format(rootpath = rootpath, path = path,temp=temp)
        cmd="slurmRun --slurm=True --cmd='python evalModel.py --msdev=True  --model_name={rootpath}/{path} --log_dir={rootpath}/{path}/Eval' --template={temp} --outputDir={rootpath}/{path}/Eval".format(rootpath = rootpath, path = path,temp=temp)
        print(cmd)
        
        os.system(cmd)
        time.sleep(30)


rootpath="/home/collab/u1368791/largefiles/TaoFiles/sentence-transformers/examples/training/ms_marco/output/biencoder-vali/"
subdir=["epoch-33steps--1-Score0.5761924603174603"]
# subdir=["124800"]
# subdir=["200000","220000","230000","235740","210000"]
# subdir=os.listdir(path=rootpath)
for path in subdir:
    curFolder=os.path.join(rootpath,path)
    if os.path.isdir(curFolder):
        # cmd="slurmRun --cmd='python eval.py  --model_name={rootpath}/{path} --log_dir={rootpath}/{path}/Eval' --template={temp} --outputDir={rootpath}/{path}/Eval".format(rootpath = rootpath, path = path,temp=temp)
        cmd="slurmRun --slurm=True --cmd='python evalModel.py --msdev=True  --model_name={rootpath}/{path} --log_dir={rootpath}/{path}/Eval' --template={temp} --outputDir={rootpath}/{path}/Eval".format(rootpath = rootpath, path = path,temp=temp)
        print(cmd)
        
        os.system(cmd)
        time.sleep(30)
        
        

temp="scripts/slurmtemp.slurm"

model_names=["msmarco-bert-base-dot-v5","msmarco-distilbert-dot-v5"]

for model_name in model_names:
    # curFolder=os.path.join(rootpath,path)
    # if os.path.isdir(curFolder):
        # cmd="slurmRun --cmd='python eval.py  --model_name={rootpath}/{path} --log_dir={rootpath}/{path}/Eval' --template={temp} --outputDir={rootpath}/{path}/Eval".format(rootpath = rootpath, path = path,temp=temp)
    cmd="slurmRun --slurm=True --cmd='python evalModel.py --msdev=True  --model_name={model_name} --log_dir=output/{model_name}/Eval' --template={temp} --outputDir=output/{model_name}/Eval".format(model_name = model_name,temp=temp)
    print(cmd)
    
    os.system(cmd)
    time.sleep(60)