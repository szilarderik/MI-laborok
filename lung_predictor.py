import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators.MLE import MaximumLikelihoodEstimator
from pgmpy.inference.ExactInference import VariableElimination

def predict(data_train: pd.DataFrame, data_test: pd.DataFrame) -> np.array:
    ######################################################
    # TODO: create a Bayesian network, learn the parameters and return the probabilities
    #       of the 'illness_yes' variable-value pair for the test data.
    # ...
    ######################################################

    # Defining the Bayesian Network structure
    asia_bnet = BayesianNetwork([
        ("asia", "tub"),
        ("smoke", "lung"),
        ("smoke", "bronc"),
        ("tub", "illness"),
        ("lung", "illness"),
        ("illness", "xray"),
        ("illness", "dysp"),
        ("bronc", "dysp")
    ])

    # Learning the parameters of the network from the training data
    asia_bnet.fit(data_train, estimator=MaximumLikelihoodEstimator)

    # Initializing the inference engine
    inference = VariableElimination(asia_bnet)

    # Iterating over the test data to calculate the probability of "illness" being "yes"
    probabilities_illness_yes = []
    for evidence in data_test.to_dict(orient="records"):
        phi_query = inference.query(variables=["illness"], evidence=evidence)
        probabilities_illness_yes.append(phi_query.values[1])  # Extracting the "yes" probability


    return np.array(probabilities_illness_yes)