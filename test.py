import sentiment_analyser as analyser
print(analyser.sentiment("This was really a horrible movie. "))
print(analyser.sentiment("Good going on baby. "))
print(analyser.sentiment("Never watched such an awesome movie ever, great acting skills, favourite movie from now"))

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""

class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = ascii(all_data["text"])
            sentiment_value, confidence = analyser.sentiment(tweet)
            print(tweet, sentiment_value, confidence*100)

            if confidence*100 >= 80:
                output = open("twitter-out.txt","a")
                output.write(sentiment_value)
            output.write('\n')
            output.close()

            return True
        except:
            return False

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["awesome"])
