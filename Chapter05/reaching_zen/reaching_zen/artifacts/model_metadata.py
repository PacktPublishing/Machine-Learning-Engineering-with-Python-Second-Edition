# MIT License
# 
# Copyright (c) Andy McMahon 2023
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
from typing import Any, Dict
from sklearn.base import ClassifierMixin

class ModelMetadata:
    """A custom artifact that stores model metadata.
    
    A model metadata object gathers together information that is collected
    about the model being trained in a training pipeline run. This data type
    is used for one of the artifacts returned by the model evaluation step.

    This is an example of a *custom artifact data type*: a type returned by
    one of the pipeline steps that isn't natively supported by the ZenML
    framework. Custom artifact data types are a common occurrence in ZenML,
    usually encountered in one of the following circumstances:
    
      - you use a third party library that is not covered as a ZenML integration
      and you model one or more step artifacts from the data types provided by
      this library (e.g. datasets, models, data validation profiles, model
      evaluation results/reports etc.)
      - you need to use one of your own data types as a step artifact and it is
      not one of the basic Python artifact data types supported by the ZenML
      framework (e.g. str, int, float, dictionaries, lists, etc.)
      - you want to extend one of the artifact data types already natively
      supported by ZenML (e.g. pandas.DataFrame or sklearn.ClassifierMixin)
      to customize it with your own data and/or behavior. 

    In all above cases, the ZenML framework lacks one very important piece of
    information: it doesn't "know" how to convert the data into a format that
    can be saved in the artifact store (e.g. on a filesystem or persistent
    storage service like S3 or GCS). Saving and loading artifacts from the
    artifact store is something called "materialization" in ZenML terms and
    you need to provide this missing information in the form of a custom
    materializer - a class that implements loading/saving artifacts from/to
    the artifact store. Take a look at the `materializers` folder to see how a
    custom materializer is implemented for this artifact data type.
    
    More information about custom step artifact data types and ZenML
    materializers is available in the docs:

      https://docs.zenml.io/advanced-guide/pipelines/materializers

    """

    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    def __init__(self) -> None:
        self.metadata: Dict[str, Any] = {}

    def collect_metadata(
        self,
        model: ClassifierMixin,
        train_accuracy: float,
        test_accuracy: float,
    ) -> None:
        """Gathers and stores metadata about a model.
        
        Args:
            model: trained model
            train_accuracy: model accuracy measured on the train set
            test_accuracy: model accuracy measured on the test set
        """
        self.metadata = dict(
            model_type = model.__class__.__name__,
            train_accuracy = train_accuracy,
            test_accuracy = test_accuracy,
        )
    
    def print_report(self) -> None:
        """Print a user-friendly report from the model metadata."""
        print(f"""
Model type: {self.metadata.get('model_type')}
Accuracy on train set: {self.metadata.get('train_accuracy')}
Accuracy on test set: {self.metadata.get('test_accuracy')}
""")
    ### YOUR CODE ENDS HERE ###
