from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score

class Evaluate_F1(evaluate):
    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        return f1_score(y_true, y_pred, average='macro', zero_division=0)