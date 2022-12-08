from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
import numpy as np
import os
import csv
from sentence_transformers.util import cos_sim, dot_score
import torch
from sklearn.metrics import average_precision_score
import tqdm

logger = logging.getLogger(__name__)
class RerankingEvaluatorMRR(evaluation.RerankingEvaluator):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("RerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)


        scores = self.compute_metrices(model)
        mean_ap = scores['map']
        mean_mrr = scores['mrr']

        #### Some stats about the dataset
        num_positives = [len(sample['positive']) for sample in self.samples]
        num_negatives = [len(sample['negative']) for sample in self.samples]

        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(len(self.samples), np.min(num_positives), np.mean(num_positives),
                                                                                                                                             np.max(num_positives), np.min(num_negatives),
                                                                                                                                             np.mean(num_negatives), np.max(num_negatives)))
        logger.info("MAP: {:.2f}".format(mean_ap * 100))
        logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr * 100))

        #### Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr])

        return mean_mrr