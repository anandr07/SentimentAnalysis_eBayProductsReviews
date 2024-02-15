#%%
#NAIVE BAYE's
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import sparse
import numpy as np

def NaiveBayes_train_simple_cv(X_train, Y_train, X_test, Y_test):
    # Split the training data for cross-validation
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    # Initialize the classifier
    nb = MultinomialNB()

    # Train the classifier
    nb.fit(X_tr, Y_tr)

    # Predict probabilities for CV and training sets
    probs_cv = nb.predict_proba(X_cv)
    probs_train = nb.predict_proba(X_tr)

    # Calculate AUC score for CV and training sets
    auc_score_cv = roc_auc_score(Y_cv, probs_cv, multi_class='ovr')
    auc_score_train = roc_auc_score(Y_tr, probs_train, multi_class='ovr')

    # Calculate accuracy for CV and training sets
    accuracy_cv = accuracy_score(Y_cv, nb.predict(X_cv))
    accuracy_train = accuracy_score(Y_tr, nb.predict(X_tr))

    print(f"AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
    print(f"AUC Score (Train): {auc_score_train}  Accuracy (Train): {accuracy_train}")

    # Confusion Matrix for CV set
    cm = confusion_matrix(Y_cv, nb.predict(X_cv))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Train')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Confusion Matrix for test set
    prob_test = nb.predict_proba(X_test)
    cm = confusion_matrix(Y_test, nb.predict(X_test))

    # Use ConfusionMatrixDisplay for visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test))
    disp.plot()
    plt.title('Confusion Matrix Test')
    plt.show()  

    # Calculate and print AUC score for test set
    auc_score = roc_auc_score(Y_test, prob_test, multi_class='ovr')
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy for test set
    accuracy = accuracy_score(Y_test, nb.predict(X_test))
    print(f"Accuracy (Test): {accuracy}")

    return auc_score, accuracy

# %%
