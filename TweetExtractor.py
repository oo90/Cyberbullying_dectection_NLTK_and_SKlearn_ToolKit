

# import necessary modules
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# import user credentials from other python file 
import twitter_credentials

tweets = []

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list, max_tweets):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename, max_tweets)
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class StdOutListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    tweet_number = 0 
    
    def __init__(self, fetched_tweets_filename, max_tweets):
        super(StdOutListener, self).__init__()
        self.fetched_tweets_filename = fetched_tweets_filename
        self.max_tweets = max_tweets


    def on_status(self, status):
        """
        This function takes only english-written tweets from twitter and saves 
        them in file.
        Parameters
        ----------
        status : TYPE
            DESCRIPTION : the R/W status of twitter API.

        Returns
        -------
        bool
            DESCRIPTION: If the certain number of tweets are extracted, then
                            return false.

        """
        try:
            with open(self.fetched_tweets_filename, 'a') as tf:     ## open file to save tweets
                if status.lang == 'en':                             ## only extracting Enlglish written tweets 
                    tf.write(status.text)
                    tf.write('\n')
                    self.tweet_number += 1                          
                         
            
        except BaseException:
            print('Error')
            pass
        
        if self.tweet_number>=self.max_tweets:
            tf.close()
            return False
            
          

    def on_error(self, status):
        print ("Error " + str(status))
        if status == 420:
            print("Rate Limited")
            return False

 
if __name__ == '__main__':
 
    # the keywords to extract tweets from twitter
    hash_tag_list = ["refugees","immigrants","islam","muslim", "gay", "bitch", "slag", "homo", "dike", "queer","boobs", "titty"]
    
    # the file to save extracted tweets
    fetched_tweets_filename = "tweets1.txt"
    
    # the number of tweets we want to extract : 8000
    max_tweets = 8000
    
    # twitter API instantiation
    twitter_streamer = TwitterStreamer()
    
    # run tweets extracting module
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list, max_tweets)
