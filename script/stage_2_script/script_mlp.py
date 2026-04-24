from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_F1 import Evaluate_F1
from code.stage_2_code.Evaluate_Precision import Evaluate_Precision
from code.stage_2_code.Evaluate_Recall import Evaluate_Recall

import numpy as np


# -------------------------
# MLP Script (Stage 2)
# -------------------------
if __name__ == "__main__":

    np.random.seed(1)

    # 1. Dataset
    data_obj = Dataset_Loader('mnist', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    # 2. Model (MLP using PyTorch)
    method_obj = Method_MLP('mlp', '')

    # 3. Result saver
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/'
    result_obj.result_destination_file_name = 'mlp_result'

    # 4. Evaluation
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # 5. Setting (IMPORTANT: NO K-FOLD)
    setting_obj = Setting_Train_Test_Split('train test split', '')

    print("************ Start ************")

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    mean_score, result= setting_obj.load_run_save_evaluate()

    data = {
        'pred_y': result['pred_y'],
        'true_y': result['true_y']
    }

    f1_eval = Evaluate_F1('f1', '')
    precision_eval = Evaluate_Precision('precision', '')
    recall_eval = Evaluate_Recall('recall', '')

    f1_eval.data = data
    precision_eval.data = data
    recall_eval.data = data

    print("F1:", f1_eval.evaluate())
    print("Precision:", precision_eval.evaluate())
    print("Recall:", recall_eval.evaluate())

    print("************ Overall Performance ************")
    print("MLP Accuracy:", mean_score)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(result['loss_list'])
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("/Users/kayyayay/Downloads/ECS189G_Winter_2022_Source_Code_Template/result/stage_2_result/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(result['acc_list'])
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("/Users/kayyayay/Downloads/ECS189G_Winter_2022_Source_Code_Template/result/stage_2_result/accuracy_curve.png")
    plt.close()

    print("************ Finish ************")