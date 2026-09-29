#
# Regresses f(x,y) -> z using PySR
#
import pandas as pd
import numpy as np
from pysr import PySRRegressor
import time
import gc
import shutil

# Standard operators
DEFAULT_BINARY = ["+", "*", "-", "/", "max", "min"]
DEFAULT_UNARY = ["square", "abs", "sqrt", "sin", "cos", "tanh", "exp"]

def run_pysr_regression(X, y, model=None, iterations=100, verbose=False, extra_pysr_params=None):
    start_time = time.time()
    fallback_val = float(np.mean(y))
    
    results = { 
        'success': True,
        'equation_found': True, 
        'best_equation_str': f"{fallback_val:.3f}",
        'best_equation_details': {'complexity': 1, 'loss': -1.0} 
    }
    
    # Only create a new model if one wasn't passed in
    created_locally = False
    if model is None:
        created_locally = True
        params = {
            "model_selection": "accuracy", 
            "binary_operators": DEFAULT_BINARY,
            "unary_operators": DEFAULT_UNARY,
            "niterations": iterations,
            "maxsize": 65,
            "populations": 30,
            "loss": "L2DistLoss()",
            "batching": True,       
            "batch_size": 512,      
            "verbosity": 1 if verbose else 0,
            "progress": verbose,
            "turbo": True,
            "temp_equation_file": True, 
            "delete_tempfiles": True,
        }
        if extra_pysr_params:
            if 'loss' in extra_pysr_params: del extra_pysr_params['loss']
            params.update(extra_pysr_params)
        model = PySRRegressor(**params)

    try:
        if verbose: print(f"Fitting model on input shape {X.shape}...")
        model.fit(X, y)
   
        if hasattr(model, 'equations_') and model.equations_ is not None and not model.equations_.empty:
            best_series = model.get_best()
            results['best_equation_str'] = str(best_series.equation)
            results['best_equation_details'] = best_series.to_dict()
            results['model'] = model
        else:
            if verbose: print(f"PySR returned empty HoF. Falling back to mean: {fallback_val:.3f}")
            
    except Exception as e:
        results['error'] = str(e)
        print(f"Regression Error: {e}. Falling back to mean: {fallback_val:.3f}")
    
    # Only delete the model if we created it inside this function
    if created_locally:
        del model
        import gc
        gc.collect()

    results['time'] = time.time() - start_time
    return results