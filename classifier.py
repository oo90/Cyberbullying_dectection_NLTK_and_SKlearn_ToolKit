# -*- coding: utf-8 -*-
import sklearn
from sklearn import linear_model, naive_bayes, svm

class Models:
    """
    class for creating, training, storing and returning classifiers.
    """
    
    def train_model(self, classifier, feature_vector_train, label):
        
        # training the classifier with training dataset
        
        newClassifier = classifier.fit(feature_vector_train, label)
        
        return newClassifier
    
    def check_model_with_sent(self, sent, classifier, featureGetter):
        """
        Check the classifier model with user-input sentences.
        """
        example = []
       
        example.append(sent)
        
        # getting features with feature_Getter
        example_features = featureGetter.transform(example)
        
        # predicting result from classifier models
        result = classifier.predict(example_features)
        
        if result[0] == 1:
            return "Cyberbullying detected!"
        else:
            return "Non Cyberbullying."

        
    def get_scores_from(self, trained_classifiers, xvalid, yvalid, scoretype):
        """
        return F1 or accuracy scores of trained classifiers according to user input
        """
        predictions = [trained_classifier.predict(xvalid) for trained_classifier in trained_classifiers]
        
        if(scoretype == "f1"):
            
            return [sklearn.metrics.f1_score(prediction, yvalid) for prediction in predictions]
        
        elif (scoretype == "accuracy"):
            
            return [sklearn.metrics.accuracy_score(prediction, yvalid) for prediction in predictions]
        
        else:
            return "Sorry, your scoretype is incorrect."
        
    def get_trained_models_from(self, xtrain, ytrain ):
        
        """
        return the list of trained classifier models
        """
        
        trained_classifiers = []
        
        trained_classifiers.append(self.train_model( naive_bayes.MultinomialNB(), xtrain, ytrain) )
                
        trained_classifiers.append(self.train_model(linear_model.LogisticRegression(max_iter = 5000), xtrain, ytrain))
        
        trained_classifiers.append(self.train_model(svm.SVC(), xtrain, ytrain))
                
        return trained_classifiers
   
    def get_confusion_matrix_from(self, trained_classifiers, xvalid, yvalid):
        """
        return confusion matrix from classifier and test data
        """
        predictions = [trained_classifier.predict(xvalid) for trained_classifier in trained_classifiers]
        
        return [sklearn.metrics.confusion_matrix(yvalid, prediction) for prediction in predictions]
