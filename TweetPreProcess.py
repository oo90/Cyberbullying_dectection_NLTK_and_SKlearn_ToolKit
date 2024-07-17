# -*- coding: utf-8 -*-

import re, nltk
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import wordnet 
from nltk.stem import WordNetLemmatizer
       
class PreProcessTweets:
    """
    Class for cleaning and preprocessing tweets.
    """
    def __init__(self):
        
        ##  import customized abbreviations replacement from external files
        abbrs = []    
        with open("abbr.txt","r") as tf:
           for line in tf:
               if not line.strip(): continue
               abbrs.append(line.strip('\n'))     
        tf.close()      
        
        ## import customized slangs replacement from external files
        slangs = []
        with open("slang.txt","r") as tf:
            for line in tf:
                if not line.strip(): continue
                slangs.append(line.strip('\t\n'))     
        tf.close()   
        
        
        self._slangs = slangs
        #self._stopwords = set(stopwords.words('english') + list(punctuation))
        self._stopwords = set(abbrs + list(punctuation))  ## define customized stopwords from abbreviations and punctuations
        #self._words = set(nltk.corpus.words.words())
        self._badwords = []
    
    def processTweets(self, list_of_tweets, is_pos_used):
        """
        This function is to return preprocessed tweets 

        Parameters
        ----------
        list_of_tweets : TYPE
            DESCRIPTION.

        Returns
        -------
        processedTweets : TYPE
            DESCRIPTION.

        """
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append(self._cleanTweet(tweet, is_pos_used))   ## call cleantweet module
        return processedTweets
    
    def _cleanTweet(self, tweet, is_pos_used):
        """
        This function is to take one tweet string and cleans it.

        Parameters
        ----------
        tweet : string
            DESCRIPTION.

        Returns
        -------
        preprocessed tweet: string

        """
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)  # remove URLs
        
        tweet = re.sub('@[^\s]+', ' ', tweet) # remove usernames
        
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        
        tweet = tweet.split()
        
        # replacing slangs with corresponding word
        for w in tweet:
            if w in self._slangs:
                ind = self._slangs.index(w)
                w = self._slangs[ind + 1]
                
        # getting rid of stopwords
        tweet = [w for w in tweet if not w in self._stopwords]
        
        tweet = " ".join(tweet)
        
        tweet = tweet.lower() # convert text to lower-case
        
        tweet = re.sub("[^a-zA-Z]+"," ",tweet)   ## letters only
        
        tweet = word_tokenize(tweet)            ## word tokenization
        
        tweet = [w for w in tweet if not w in self._stopwords]
        
        # word lemmatization
        lemmatizer = WordNetLemmatizer()
        
        if is_pos_used:
            
            tweet = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tweet] + [v for k,v in nltk.pos_tag(tweet)]
            
        else:
            
            tweet = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tweet]
        
        return tweet
    
    def get_wordnet_pos(self, word):
        
        """Map POS tag to first character lemmatize() accepts"""
        
        tag = nltk.pos_tag([word])[0][1][0].upper()
        
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "R": wordnet.ADV}
    
        return tag_dict.get(tag, wordnet.NOUN)
    
    def labeling(self, tweets):
        """
        This function is to take row of tweets and labels them as bullying or noncyberbullying

        Parameters
        ----------
        tweets : list of string
            DESCRIPTION.

        Returns
        -------
        result : list of dictionary containing tweet and label
            DESCRIPTION.

        """
        result = [] 
        
        self.get_badwords_list()
        ## check if the tweet contain the words from badwords list
        ## if extsts, then label as cyberbullying
        ## else, label the tweet as non-cyberbullying
        for tweet in tweets:
            flag = 0 
            if( len(tweet) == 0):continue
            for word in tweet:
                if word in self._badwords:
                    result.append([tweet, 1]) # cyberbullying 
                    flag = 1
                    break
            if flag == 0:  
                result.append([tweet, 0])        # non-cyberbullying
        return result
    
    def get_badwords_list(self):
        
        ## make badwords list from external file
        for line in open("badwords.txt"):
            for word in line.split( ):
                self._badwords.append(word)


