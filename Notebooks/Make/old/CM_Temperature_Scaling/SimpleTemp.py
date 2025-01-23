import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

def _row_max_normalization(data: np.ndarray) -> np.ndarray:
    '''Normalise the output by subtracting
       the per-row maximum element.
    '''
    row_max: np.ndarray = np.max(data,   axis = 1,  keepdims = True )
    
    return data - row_max

def _softmax_T(predictions: np.ndarray, 
               temperature: float,
              ) -> np.ndarray:
    '''Softmax function scaled by the
       inverse temperature.
    '''
    
    softmax_T_output: np.ndarray = predictions
    softmax_T_output = _row_max_normalization(softmax_T_output)  
    softmax_T_output /= temperature  
    softmax_T_output = softmax(softmax_T_output, 
                               axis = 1
                              )
    softmax_T_output = softmax_T_output.astype(dtype = predictions.dtype)
    
    return softmax_T_output

def _exp_T(predictions: np.ndarray, 
           temperature: float
          ) -> np.ndarray:
    '''Scale by inverse temperature,
       and then apply the nature
       exponential function
    '''
    
    exp_T_output: np.ndarray = predictions
    exp_T_output = _row_max_normalization(exp_T_output)
    exp_T_output /= temperature
    exp_T_output = np.exp(exp_T_output)
    
    return exp_T_output 

def _temperature_scaling(predictions: np.ndarray, 
                         labels: np.ndarray, 
                         initial_temperature: float
                        ) -> float:
    
    def negative_log_likelihood(temperature: float):
        '''Negative Log Likelihood Loss with respect
           to Temperature
        '''
        
        # Losses
        losses: np.ndarray = _softmax_T(predictions, 
                                          temperature
                                         )
            
        # Select the probability of the correct class
        losses = losses[np.arange(losses.shape[0]), 
                        labels
                       ]
        
        losses = np.log(losses)
        
        # Derivates with respect to Temperature
        exp_T: np.ndarray = _exp_T(predictions, temperature)
        exp_T_sum = exp_T.sum(axis = 1)
        
        term_1: np.ndarray = _row_max_normalization(predictions)
        term_1 /= temperature ** 2
        term_1 = - term_1[np.arange(term_1.shape[0]), 
                          labels
                         ]
        term_1 *= exp_T_sum
        
        
        
        term_2: np.ndarray = _row_max_normalization(predictions)
        term_2 /= temperature ** 2
        term_2 = _row_max_normalization(term_2)
        term_2 *= exp_T
        term_2 = term_2.sum(axis = 1)
        
        dL_dts: np.ndarray = (term_1 + term_2) / exp_T_sum
            
        # print(f"{-losses.sum() = },  {-dL_dts.sum() = }")
            
        return -losses.sum(),  -dL_dts.sum()
    
    temperature_minimizer: minimize = minimize(negative_log_likelihood, 
                                               initial_temperature, 
                                               method = "L-BFGS-B",
                                               jac = True,
                                               options = {"gtol": 1e-6,
                                                           "ftol": 64 * np.finfo(float).eps,
                                                         }
                                              )
        
    return temperature_minimizer.x[0]