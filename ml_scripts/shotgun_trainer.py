from sklearn.model_selection import cross_validate
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

class try_all_models():
    
    def __init__(self, models, cv_params):
        self.models = models
        self.cv_params = cv_params

        self.train_scores = []
        self.val_scores = []
        self.model_order = []

        self.data_dict = {"model":[],"score":[],"dataset":[]}

    def fit(self, X, y):
        for model_i in self.models:
            cv_results = cross_validate(estimator=model_i, X=X, y=y, **self.cv_params)
            try:
                self.data_dict["score"] += cv_results['train_score'].tolist() + cv_results['test_score'].tolist()
                self.data_dict["dataset"] += ["train" for _ in cv_results['train_score'].tolist()] + ["val" for _ in cv_results['test_score'].tolist()]
                self.data_dict["model"] += [type(model_i).__name__ for _ in cv_results['train_score'].tolist() + cv_results['test_score'].tolist()]
            except Exception:
                for score_name in self.cv_params['scoring']:
                    self.data_dict["score"] += cv_results['train_' + score_name].tolist() + cv_results['test_' + score_name].tolist()
                    self.data_dict["dataset"] += ["train" for _ in cv_results['train_' + score_name].tolist()] + ["val" for _ in cv_results['test_' + score_name].tolist()]
                    self.data_dict["model"] += [type(model_i).__name__ for _ in cv_results['train_' + score_name].tolist() + cv_results['test_' + score_name].tolist()]
            
        
    def plot(self):
        '''
        # Set Font Size
        SMALL_SIZE = 18
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 18

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)
        '''
        
        pass

    def _build_df(self):
        # train_scores is a list of numpy arrays, where each element in the list corresponds to a models scores
        # val_scores is a list of numpy arrays, where each element in the list corresponds to a models scores
        # model_order is a list of model names
        
        '''
        TODO: Unpack data from lists to create a single row for each loss/score for each of the cv runs
        TODO: Create dataframe with template ["model","loss","]
        '''
        # Unpack data into dictionary
        pass

if __name__ == "__main__":
    print("hello")
