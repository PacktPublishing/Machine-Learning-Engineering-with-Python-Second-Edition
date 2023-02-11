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

from zenml.pipelines import pipeline


@pipeline()
def model_training_pipeline(
    data_loader,
    data_processor,
    data_splitter,
    model_trainer,
    model_evaluator,
):
    """
    Model training pipeline recipe.

    This is a recipe for a pipeline that loads the data, processes it and
    splits it into train and test sets, then trains and evaluates a model
    on it. It is agnostic of the actual step implementations and just defines
    how the artifacts are circulated through the steps by calling them in the
    right order and passing the output of one step as the input of the next
    step.

    The arguments that this function takes are instances of the steps that
    are defined in the steps folder. Also note that the arguments passed to
    the steps are step artifacts. If you use step parameters to configure the
    steps, they must not be used here, but instead be used when the steps are
    instantiated, before this function is called.

    Args:
        data_loader: A data loader step instance that outputs a dataset.
        data_processor: A data processor step instance that takes a dataset
            as input and outputs a processed dataset.
        data_splitter: A data splitter step instance that takes a dataset
            as input and outputs a train and test set split.
        model_trainer: A model trainer step instance that takes a train set
            as input and outputs a trained model.
        model_evaluator: A model evaluator step instance that takes a train/test
            set split and a trained model as input and outputs model evaluation
            metrics.

    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    dataset = data_loader()
    processed_dataset = data_processor(dataset=dataset)
    train_set, test_set = data_splitter(dataset=processed_dataset)
    model = model_trainer(train_set=train_set)
    model_evaluator(
        model=model,
        train_set=train_set,
        test_set=test_set,
    )
    ### YOUR CODE ENDS HERE ###
