from sklearn.model_selection import cross_validate
import seaborn as sns
from matplotlib import pyplot as plt

class try_all_models():
    
    def __init__(self, models, cv_params):
        self.models = models
        self.cv_params = cv_params

        self.train_scores = []
        self.val_scores = []
        self.model_order = []

    def fit(self, X, y):
        for model_i in self.models:
            cv_results = cross_validate(estimator=model_i, X=X, y=y, **self.cv_params)
            try:
                self.train_scores.append(cv_results['train_score'])
                self.val_scores.append(cv_results['test_score'])
            except Exception:
                self.train_scores.append([cv_results['train_' + score_name] for score_name in self.cv_params['scoring']])
                self.val_scores.append([cv_results['test_' + score_name] for score_name in self.cv_params['scoring']])
            self.model_order.append(type(model_i).__name__)
        
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