from sentence_transformers.cross_encoder import CrossEncoder
import os
class CrossEncoderSave(CrossEncoder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(os.path.join(output_path,str(steps)))