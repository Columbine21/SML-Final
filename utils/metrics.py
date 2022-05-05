import scipy as sp
import numpy as np
import pandas as pd

def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score

class MetricsTop():
    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_regression(self, y_pred, y_true):

        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        # Average L1 distance between preds and truths
        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        test_truth_a5 = np.round(test_truth * 4)
        test_preds_a5 = np.round(test_preds * 4)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        eval_results = {
            "Acc-5": round(mult_a5, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results

    def getMetics(self):
        return self.__eval_regression