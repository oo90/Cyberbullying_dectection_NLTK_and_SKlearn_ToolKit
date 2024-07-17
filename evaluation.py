# -*- coding: utf-8 -*-

from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import csv
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for dumping ready classifiers

import joblib

from TweetPreProcess import PreProcessTweets
import classifier


count_trained_classifiers, word_tfidf_trained_classifiers, bigram_tfidf_trained_classifiers, triigram_tfidf_trained_classifiers = [], [], [], []


def clean_tweet_and_save(orginFileName, cleanFileName, is_pos_used):
    # taking in the dataset from external files
    tweets = []

    with open(originFileName, "r") as tf:
        for line in tf:
            if not line.strip():
                continue
            tweets.append(line.strip('\n'))

    tf.close()

    # instantiation of preprocessing tweet class
    # and perform preprocessing tweets
    tweetProcessor = PreProcessTweets()

    cleaned_tweets = tweetProcessor.processTweets(tweets, is_pos_used)

    labeled_tweets = tweetProcessor.labeling(cleaned_tweets)

    # save cleaned__tweets in 'cleaned_tweets.csv' file
    myFile = open(cleanFileName, 'w')

    with myFile:
        myFields = ['tweet', 'cyberbullying']
        writer = csv.DictWriter(myFile, fieldnames=myFields)
        writer.writeheader()
        for tweet, label in labeled_tweets:
            writer.writerow({myFields[0]: " ".join(tweet), myFields[1]: label})
    myFile.close()

    return cleanFileName


def comparison_boxfigure(names, results, score_type):
    fig = plt.figure()
    if score_type == "F1":
        fig.suptitle('F1-score Comparison')
    else:
        fig.suptitle('Accuracy Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def comparison_barfigure(results, score_type):
    ######   Accuracy/F1-score Comparison of algorithms in various feature selectors   #####
    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars1 = results[0]
    bars2 = results[1]
    bars3 = results[2]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth,
            edgecolor='white', label='NB')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth,
            edgecolor='white', label='LR')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth,
            edgecolor='white', label='SVM')

    # Add xticks on the middle of the group bars
    plt.xlabel(score_type + ' (feature set)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], features)

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def cleaned_to_dataframe(csvFileName):

    trainDF = pd.read_csv(csvFileName)

    # size manipulation: ratio of cyberbullying to non_cyberbullying as 60:40
    cyberbullying = trainDF[trainDF['cyberbullying'] == 1]

    non_cyberbullying = trainDF[trainDF['cyberbullying'] == 0]

    newsize = (int)(non_cyberbullying.shape[0] * (6/4))

    new_cyberbullying = cyberbullying.sample(n=newsize)

    new_trainDF = pd.concat(
        [new_cyberbullying, non_cyberbullying]).sort_index()

    return new_trainDF

    print(new_trainDF)


def dataset_preparation(cleanedFileName, features=0):

    dataset = {}

    feature_getters = []

    trainDF = cleaned_to_dataframe(cleanedFileName)

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
        trainDF['tweet'], trainDF['cyberbullying'])

    # # label encode the target variable
    # encoder = preprocessing.LabelEncoder()
    # train_y = encoder.fit_transform(train_y)
    # valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer object
    if features != 0:
        count_vect = CountVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', max_features=features)
        tfidf_vect = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', max_features=features)
        tfidf_vect_bigram = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 2), max_features=features)
        tfidf_vect_trigram = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(3, 3), max_features=features)
    else:
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        tfidf_vect_bigram = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 2))
        tfidf_vect_trigram = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(3, 3))

    count_vect.fit(trainDF['tweet'])
    tfidf_vect.fit(trainDF['tweet'])
    tfidf_vect_bigram.fit(trainDF['tweet'])
    tfidf_vect_trigram.fit(trainDF['tweet'])

    feature_getters.append(count_vect)
    feature_getters.append(tfidf_vect)
    feature_getters.append(tfidf_vect_bigram)
    feature_getters.append(tfidf_vect_trigram)

    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(train_x)

    xvalid_count = count_vect.transform(valid_x)

    dataset['xtrain_count'] = xtrain_count

    dataset['xvalid_count'] = xvalid_count

    # unigram tf-idf
    # transform the training and validation data using unigram + tfidf vectorizer object
    xtrain_tfidf = tfidf_vect.transform(train_x)

    xvalid_tfidf = tfidf_vect.transform(valid_x)

    dataset['xtrain_tfidf'] = xtrain_tfidf

    dataset['xvalid_tfidf'] = xvalid_tfidf

    # bigram + tf-idf

    # transform the training and validation data using bigram + tfidf vectorizer object
    xtrain_tfidf_bigram = tfidf_vect_bigram.transform(train_x)

    xvalid_tfidf_bigram = tfidf_vect_bigram.transform(valid_x)

    dataset['xtrain_tfidf_bigram'] = xtrain_tfidf_bigram

    dataset['xvalid_tfidf_bigram'] = xvalid_tfidf_bigram

    # trigram + tf-idf

    # transform the training and validation data using trigram + tfidf vectorizer object
    xtrain_tfidf_trigram = tfidf_vect_trigram.transform(train_x)

    xvalid_tfidf_trigram = tfidf_vect_trigram.transform(valid_x)

    dataset['xtrain_tfidf_trigram'] = xtrain_tfidf_trigram

    dataset['xvalid_tfidf_trigram'] = xvalid_tfidf_trigram

    dataset['train_y'] = train_y

    dataset['valid_y'] = valid_y

    return dataset, feature_getters


def get_feature_best_number(matrix):
    """
    take the feature vectors of training set as input
    and output the number of optimally selected features

    Parameters
    ----------
    matrix : TYPE => pandas dataframe
        DESCRIPTION.
            feature vector of training set
    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    Example: size =  get_feature_best_number(dataset['xtrain_count'])
            ## size = (6351, 8171)
    """
    matrix_T = matrix.T

    cos_matrix = cosine_similarity(matrix_T)

    df = pd.DataFrame(cos_matrix)

    df_masked = df.mask(np.triu(np.ones(df.shape, dtype=np.bool_)))

    df_only_val = df_masked[df_masked > 0]

    temp = df_only_val.count(axis=1)

    return temp[temp > 0].shape[0], matrix.shape[1]


if __name__ == '__main__':

    # set if is POS tag used in feature engineering
    is_pos_used = True

    # set number of features. if set as 0, then use all features
    feature_num = 5000

    ## set the name of original tweet file ##
    originFileName = "cleaned_tweets.csv"

    ## set the name of file to store cleaned tweet  ###
    cleanedFileName = "cleaned_tweets.csv"

    ## tweet preprocessing and cleaning, save cleaned tweets to .csv file  ###
    clean_tweet_and_save(originFileName, cleanedFileName, is_pos_used)

    ### obtain dataset and feature generation methods ####
    dataset, feature_getters = dataset_preparation(
        cleanedFileName, feature_num)

    ### getting know the number of optimal features for training classifier  ###
    # (optimal_feature_num, current_feature_num) = get_feature_best_number(dataset['xtrain_count'])
    # (optimal_feature_num, current_feature_num) = get_feature_best_number(dataset['xtrain_tfidf'])
    # (optimal_feature_num, current_feature_num) = get_feature_best_number(dataset['xtrain_tfidf_bigram'])
    # (optimal_feature_num, current_feature_num) = get_feature_best_number(dataset['xtrain_tfidf_trigram'])

    ###############   Get Models and Train them   ###############
    models = classifier.Models()

    ##############  Geting scores  ##############################

    ### get trained classifiers ###
    count_trained_classifiers = models.get_trained_models_from(
        dataset['xtrain_count'], dataset['train_y'])
    word_tfidf_trained_classifiers = models.get_trained_models_from(
        dataset['xtrain_tfidf'], dataset['train_y'])
    bigram_tfidf_trained_classifiers = models.get_trained_models_from(
        dataset['xtrain_tfidf_bigram'], dataset['train_y'])
    trigram_tfidf_trained_classifiers = models.get_trained_models_from(
        dataset['xtrain_tfidf_trigram'], dataset['train_y'])

    ### get Accuracy score ###
    count_accuracy_scores = models.get_scores_from(
        count_trained_classifiers, dataset['xvalid_count'], dataset['valid_y'], "accuracy")
    word_tfidf_accuracy_scores = models.get_scores_from(
        word_tfidf_trained_classifiers, dataset['xvalid_tfidf'], dataset['valid_y'], "accuracy")
    bigram_tfidf_accuracy_scores = models.get_scores_from(
        bigram_tfidf_trained_classifiers, dataset['xvalid_tfidf_bigram'], dataset['valid_y'], "accuracy")
    trigram_tfidf_accuracy_scores = models.get_scores_from(
        trigram_tfidf_trained_classifiers, dataset['xvalid_tfidf_trigram'], dataset['valid_y'], "accuracy")

    ### get F1 score ###
    count_f1_scores = models.get_scores_from(
        count_trained_classifiers, dataset['xvalid_count'],  dataset['valid_y'], "f1")
    word_tfidf_f1_scores = models.get_scores_from(
        word_tfidf_trained_classifiers, dataset['xvalid_tfidf'],  dataset['valid_y'], "f1")
    bigram_tfidf_f1_scores = models.get_scores_from(
        bigram_tfidf_trained_classifiers, dataset['xvalid_tfidf_bigram'],  dataset['valid_y'], "f1")
    trigram_tfidf_f1_scores = models.get_scores_from(
        trigram_tfidf_trained_classifiers,  dataset['xvalid_tfidf_trigram'],  dataset['valid_y'], "f1")

    ### get confusion matrix  ###
    count_conf_matrix = models.get_confusion_matrix_from(
        count_trained_classifiers, dataset['xvalid_count'],  dataset['valid_y'])
    unitfidf_conf_matrix = models.get_confusion_matrix_from(
        word_tfidf_trained_classifiers, dataset['xvalid_tfidf'],  dataset['valid_y'])
    bitfidf_conf_matrix = models.get_confusion_matrix_from(
        bigram_tfidf_trained_classifiers, dataset['xvalid_tfidf_bigram'],  dataset['valid_y'])
    tritfidf_conf_matrix = models.get_confusion_matrix_from(
        trigram_tfidf_trained_classifiers, dataset['xvalid_tfidf_trigram'],  dataset['valid_y'])

    print("\n" + str(is_pos_used) + " " + str(feature_num) + "\n")
    print("Accuracy:\n")
    print("Count Vectors: ")
    for score in count_accuracy_scores:
        print(score, "\n")

    print("Unigram TF-IDF: ")
    for score in word_tfidf_accuracy_scores:
        print(score, "\n")

    print("Bi-Gram Vectors: ")
    for score in bigram_tfidf_accuracy_scores:
        print(score, "\n")

    print("Tri-Gram Vectors: ")
    for score in trigram_tfidf_accuracy_scores:
        print(score, "\n")

    print("f1_scores:\n")
    print("Count Vectors: ")
    for score in count_f1_scores:
        print(score, "\n")

    print("Unigram TF-IDF: ")
    for score in word_tfidf_f1_scores:
        print(score, "\n")

    print("Bi-Gram Vectors: ")
    for score in bigram_tfidf_accuracy_scores:
        print(score, "\n")

    print("Tri-Gram Vectors: ")
    for score in trigram_tfidf_f1_scores:
        print(score, "\n")

    classifiers = [count_trained_classifiers, word_tfidf_trained_classifiers,
                   bigram_tfidf_trained_classifiers, trigram_tfidf_trained_classifiers]
    acc_scores = [count_accuracy_scores, word_tfidf_accuracy_scores,
                  bigram_tfidf_accuracy_scores, trigram_tfidf_accuracy_scores]
    f_scores = [count_f1_scores, word_tfidf_f1_scores,
                bigram_tfidf_f1_scores, trigram_tfidf_f1_scores]

    ####   dump the best classifier to external pickle file   #####
    save_Classifier = "pickles/Classifier.pickle"
    joblib.dump(classifiers[1][2], save_Classifier)  # SVM unigram tfidf

    save_featureGetters = "pickles/featureGetters.pickle"
    joblib.dump(feature_getters[1], save_featureGetters)  # unigram tfidf  ###

    #####   checking models with sentences    #####
    sent = "peter is the fat bitch at work place"
    check_results = []
    for featureindex in range(4):
        for modelindex in range(3):
            check_results.append(models.check_model_with_sent(
                sent, classifiers[featureindex][modelindex], feature_getters[featureindex]))

    ###### visualising accuracy of various models and features   ##################

    nb_accuracy, lr_accuracy, svm_accuracy = [], [], []

    nb_f, lr_f, svm_f = [], [], []

    for score in acc_scores:
        nb_accuracy.append(score[0])
        lr_accuracy.append(score[1])
        svm_accuracy.append(score[2])

    for score in f_scores:
        nb_f.append(score[0])
        lr_f.append(score[1])
        svm_f.append(score[2])

    names = ['NB', 'LR', 'SVM']

    features = ['count', 'unigram tfidf', 'bigram tfidf', 'trigram tfidf']

    acc_results = [nb_accuracy, lr_accuracy, svm_accuracy]

    f_results = [nb_f, lr_f, svm_f]

    comparison_boxfigure(names, acc_results, 'Accuracy')

    comparison_boxfigure(names, f_results, 'F1')

    comparison_barfigure(acc_results, 'Accuracy comparison')

    comparison_barfigure(f_results, 'F1-score comparison')
