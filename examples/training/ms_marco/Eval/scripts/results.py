import BatchExpLaunch.results_org as rog
import pandas as pd
import os
# scriptPath=os.path.dirname(os.path.abspath(__file__))
# os.chdir(scriptPath+"/..")
path_root="../output/mse-huggingfaceHard10EpochDist"
path_root="../output/mse-huggingfaceHard10EpochCo-con"
path_root="../../output/train_bi-encoder-margin_mse-distilbert-base-uncased-batch_size_64-2022-11-30_11-40-37"
OutputPath=os.path.join(path_root,"result")
# _,MeanResult=rog.get_result_df(path_root,rerun=True)
Result,MeanResult=rog.get_result_df(path_root,groupby=None)
print(Result)