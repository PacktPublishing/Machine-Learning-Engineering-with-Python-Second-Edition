import module
import pandas as pd
import numpy as np

if __name__=="__main__":
    df = pd.DataFrame({'data': np.linspace(0, 1, 100)})
    module.calculate_statistics(df)
    module.make_func_result_json(module.calculate_statistics, df)