from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
def add_number(X, columns=None, number=None):
    if columns == None and number == None:
        return X
    X[columns] = X[columns] + number
    

    
pipeline = Pipeline(
    steps=[
        "add_float",
        FunctionTransformer(
            add_number, 
            kw_args={"columns": ["col1", "col2", "col3"], "number": 0.5}
            )
        ]
    )