#HYPERPARAMETER OPTIMISATION
from timeit import default_timer as timer
from hyperopt import STATUS_OK, Trials, fmin, tpe
from .fit import *

class fitness:
    """
    For conducting cross validation on a model with a given set of hyperparameters for optimisation.
    
    Parameters
    ----------
    model_dict : dict
        Key word arguments to be fed into a b.Model class.
    model_param_names : List
        Hyperparameter names specific to the regressor model.
    training_param_names : List
        Hyperparameter names specific to training.
    """
    def __init__(self, model_dict, model_param_names, training_param_names, datasets):
        self.m = model_dict
        self.model_param_names = model_param_names
        self.training_param_names = training_param_names
        self.datasets = datasets
    
    def objective(self, params):
        """
        Objective function for bayesian hyperparameter optimisation.
        
        Parameters
        ----------
        params : dict
            Specific set of model and training hyperparameters for testing.
        
        Returns
        -------
        dict
            Results of CV testing, including MAE loss, runtime and the original parameter list"""
        
        model_params = dict()
        training_params = dict()
        for param_name in self.model_param_names:
            model_params[param_name] = params[param_name]
        for param_name in self.training_param_names:
            training_params[param_name] = params[param_name]
        
        copy = self.m['model']
        self.m['model'] = self.m['model'](**model_params)
        self.m.update(training_params)
        
        model = Model(**self.m)
        data = self.datasets[model.data_type]

        start = timer()
        res, full_res = CV_fit(model, data)
        run_time = timer()-start

        loss = res[0]
        self.m['model'] = copy
        
        return {'loss': loss, 'params': params, 'run_time': run_time, 'status': STATUS_OK}
    
def hyperopt_func(model_dict, model_param_names, training_param_names, param_space, datasets, max_evals=30):
    """
    Bayesian hyperparameter optimisation function.
    
    Parameters
    ----------
    model_dict : dict
        Key word arguments to be fed into a b.Model class.
    model_param_names : List
        Hyperparameter names specific to the regressor model.
    training_param_names : List
        Hyperparameter names specific to training.
    param_space : dict
        Distribution of choices for each hyperparameter to be optimised.
    datasets : dict
        Dictionary of the data to be used for fitting.
    max_evals : int
        Maximum number of evaluations of hyperparameter sets.
        
    Returns
    -------
    results : list
        Results from each evaluation of the objective function, sorted from best to worst result.
    """
    tester = fitness(model_dict, model_param_names, training_param_names, datasets)
    trials = Trials()
    
    timer_start = timer()
    best = fmin(fn=tester.objective, 
                space=param_space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials, 
                rstate=np.random.RandomState(50))
    timer_end = timer()
    print('Total training time (min):',(timer_end-timer_start)/60)
    results = sorted(trials.results, key = lambda x: x['loss'])
    return results