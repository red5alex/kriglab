import pandas as pd
import numpy as np
import math as m

def sine_data(start = 0,
              stop = 3*365,
              resolution = 3*365,
              period = 365,             
              amplitude = 10,
              noise_type ="gaussian",
              noise_mean = 0,
              noise_std = 1,
              n_sample = None):

    df_z = pd.DataFrame( np.linspace(start, stop, resolution), columns=['x'] )
    
    if n_sample is not None:
        df_z = df_z.sample(n_sample)
    
    df_z["y"] = 0
    df_z["true"] = amplitude * np.sin(df_z["x"]/(365.24)*2*m.pi)
    if noise_type == "gaussian":
        df_z["noise"] = np.random.randn(len(df_z)) * noise_std + noise_mean
        df_z["confidence95"] = noise_std * 1.96
        df_z["measurement"] = df_z["true"] + df_z["noise"] 
    else:
        raise ValueError("unknown noise_type "+str(noise_type))
     
    return df_z

