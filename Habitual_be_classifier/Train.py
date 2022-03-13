from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier



def algo_trainers(X_tr_counts, y_tr_counts):
    """
    Trains all of the ML models

    Parameters
        - X_tr_counts: Array of shape [samples x features]
        - y_tr_counts: Array of shape [samples]

    Returns
        - All the trained ML algorithms

    """

    # Creates the models for LR and SGD, then makes the predictions on the test sets
    LR_model = LogisticRegression(max_iter=3000, class_weight='auto')
    LR_model.fit(X_tr_counts, y_tr_counts.ravel())

    SGD_model = SGDClassifier(max_iter=3000, tol=1e-3)
    SGD_model.fit(X_tr_counts, y_tr_counts.ravel())

    MLP_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
    MLP_model.fit(X_tr_counts, y_tr_counts.ravel())

    cf = LogisticRegression(random_state=1)
    cf2 = SGDClassifier(random_state=1)
    cf3 = MLPClassifier(random_state=1, alpha=1e-5, solver='lbfgs', hidden_layer_sizes=(5, 2), max_iter=10000)
    ensemble_model = VotingClassifier(estimators=[('lr', cf), ('sgd', cf2), ('mlp', cf3)], voting='hard')
    ensemble_model.fit(X_tr_counts, y_tr_counts.ravel())
    ML_models = {'logistic regression' : LR_model,
                'sgd' : SGD_model,
                'mlp' : MLP_model,
                'ensemble' : ensemble_model
                 }
    return ML_models
