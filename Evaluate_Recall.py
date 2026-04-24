from code.base_class.evaluate import evaluate
from sklearn.metrics import recall_score

class Evaluate_Recall(evaluate):
    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        return recall_score(y_true, y_pred, average='macro', zero_division=0)