from sklearn.model_selection import cross_validate
from sklearn.exceptions import NotFittedError
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os
from time import strftime

class try_all_models():
    
    def __init__(self, models, cv_params):
        self.models = models
        self.cv_params = cv_params

        self.data_dict = {"model":[],"score":[],"dataset":[],"metric":[]}

        self._run_id = None
        self._fit = False

    def fit(self, X, y):
        self._validate_inputs()

        self._run_id = strftime("run_%Y_%m_%d-%H_%M_%S")
        for model_i in self.models:
            cv_results = cross_validate(estimator=model_i, X=X, y=y, **self.cv_params)
            try:
                self.data_dict["score"] += (-1*cv_results['train_score']).tolist() + (-1*cv_results['test_score']).tolist()
                self.data_dict["dataset"] += ["train" for _ in cv_results['train_score'].tolist()] + ["val" for _ in cv_results['test_score'].tolist()]
                self.data_dict["model"] += [type(model_i).__name__ for _ in cv_results['train_score'].tolist() + cv_results['test_score'].tolist()]
                self.data_dict["metric"] += [self.cv_params['scoring'] for _ in cv_results['train_score'].tolist() + cv_results['test_score'].tolist()]
            except Exception:
                for score_name in self.cv_params['scoring']:
                    self.data_dict["score"] += (-1*cv_results['train_' + score_name]).tolist() + (-1*cv_results['test_' + score_name]).tolist()
                    self.data_dict["dataset"] += ["train" for _ in cv_results['train_' + score_name].tolist()] + ["val" for _ in cv_results['test_' + score_name].tolist()]
                    self.data_dict["model"] += [type(model_i).__name__ for _ in cv_results['train_' + score_name].tolist() + cv_results['test_' + score_name].tolist()]
                    self.data_dict["metric"] += [score_name for _ in cv_results['train_' + score_name].tolist() + cv_results['test_' + score_name].tolist()]
            
    def _plot(self, metric, save=False):
        self._check_fit()

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
        
        temp_df = pd.DataFrame(data=self.data_dict)
        sns.lineplot(x="model",y="score",hue="dataset",data=temp_df.loc[temp_df['metric'] == metric])
        
        if save:
            plt.savefig(self._run_id+'.png')

    def _save(self):
        self._check_fit()

        pd.DataFrame(data=self.data_dict).to_csv(self._run_id+".csv", index=False)
        try:
            self._plot(metric=self.cv_params["scoring"])
        except Exception:
            self._plot(metric=self.cv_params["scoring"][0])

    def _validate_inputs(self):
        pass

    def _check_fit(self):
        if self._fit == False:
            raise NotFittedError('Model has not been fit yet')

def main():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline  
    
    size=10000

    cv_params_single = {
        "cv":10,
        "scoring":"neg_mean_squared_error",
        "return_train_score":True
    }

    np.random.seed(12359780)
    X = np.random.uniform(low=-10,high=10,size=size).reshape((-1,1))
    y = np.power(X,2) + np.random.randn(size)

    models = [LinearRegression(),make_pipeline(PolynomialFeatures(2),LinearRegression())]
    trainer = try_all_models(models, cv_params_single)
    trainer.fit(X,y)
    
if __name__ == "__main__":
    main()
