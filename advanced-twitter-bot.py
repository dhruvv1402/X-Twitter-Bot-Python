import tweepy
import time
import logging
import random
import os
import json
import pandas as pd
import numpy as np
import schedule
import threading
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg') 
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_twitter_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedTwitterBot")

load_dotenv()


API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("ACCESS_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")


DATA_FOLDER = "bot_data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    os.makedirs(os.path.join(DATA_FOLDER, "images"))
    os.makedirs(os.path.join(DATA_FOLDER, "analytics"))

class AdvancedTwitterBot:
    """An advanced bot for Twitter automation, analytics, and content generation."""
    
    def __init__(self):
        """Initialize the bot with Twitter API authentication and NLP tools."""
        try:
            
            auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
           
            self.client = tweepy.Client(
                bearer_token=BEARER_TOKEN,
                consumer_key=API_KEY,
                consumer_secret=API_SECRET,
                access_token=ACCESS_TOKEN,
                access_token_secret=ACCESS_SECRET,
                wait_on_rate_limit=True
            )
            
          
            self.api.verify_credentials()
            logger.info("Authentication successful!")
            
           
            logger.info("Initializing NLP models...")
            self.sentiment_analyzer = TextBlob
            
            self.text_generator = pipeline('text-generation', model='gpt2')
            
            self.sentiment_classifier = pipeline('sentiment-analysis')
        
            self.tweets_data = []
            self.followers_data = []
            self.engagement_data = []
            self.last_mention_id = None
            self.topics_to_monitor = []
            self.scheduled_tweets = []
            
            self._load_data()
            
            logger.info("Bot initialization complete!")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _load_data(self):
        """Load existing data from JSON files if available."""
        try:
         
            topics_file = os.path.join(DATA_FOLDER, "topics.json")
            if os.path.exists(topics_file):
                with open(topics_file, 'r') as f:
                    self.topics_to_monitor = json.load(f)
                logger.info(f"Loaded {len(self.topics_to_monitor)} topics to monitor")
            
          
            schedule_file = os.path.join(DATA_FOLDER, "schedule.json")
            if os.path.exists(schedule_file):
                with open(schedule_file, 'r') as f:
                    self.scheduled_tweets = json.load(f)
                logger.info(f"Loaded {len(self.scheduled_tweets)} scheduled tweets")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save bot data to JSON files."""
        try:
          
            with open(os.path.join(DATA_FOLDER, "topics.json"), 'w') as f:
                json.dump(self.topics_to_monitor, f)
            
           
            with open(os.path.join(DATA_FOLDER, "schedule.json"), 'w') as f:
                json.dump(self.scheduled_tweets, f)
            
           
            if self.tweets_data:
                df = pd.DataFrame(self.tweets_data)
                df.to_csv(os.path.join(DATA_FOLDER, "analytics", f"tweets_data_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
            
            logger.info("Bot data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    
    def post_tweet(self, text, media_path=None):
        """Post a new tweet, optionally with media."""
        try:
            if media_path:
               
                media = self.api.media_upload(media_path)
                tweet = self.client.create_tweet(text=text, media_ids=[media.media_id])
            else:
                tweet = self.client.create_tweet(text=text)
            
            tweet_id = tweet.data['id']
            logger.info(f"Tweet posted successfully! ID: {tweet_id}")
            
          
            self.tweets_data.append({
                'id': tweet_id,
                'text': text,
                'created_at': datetime.now().isoformat(),
                'has_media': bool(media_path),
                'likes': 0,
                'retweets': 0,
                'replies': 0
            })
            
            return tweet_id
        except Exception as e:
            logger.error(f"Failed to post tweet: {e}")
            return None
    
    def delete_tweet(self, tweet_id):
        """Delete a tweet by ID."""
        try:
            self.client.delete_tweet(id=tweet_id)
            logger.info(f"Tweet {tweet_id} deleted successfully!")
            
           
            self.tweets_data = [t for t in self.tweets_data if t['id'] != tweet_id]
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete tweet {tweet_id}: {e}")
            return False
    
    def retweet(self, tweet_id):
        """Retweet a tweet by ID."""
        try:
            self.client.retweet(tweet_id)
            logger.info(f"Successfully retweeted tweet {tweet_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to retweet {tweet_id}: {e}")
            return False
    
    def like_tweet(self, tweet_id):
        """Like a tweet by ID."""
        try:
            self.client.like(tweet_id)
            logger.info(f"Successfully liked tweet {tweet_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to like tweet {tweet_id}: {e}")
            return False
    
    def follow_user(self, user_id):
        """Follow a user by ID."""
        try:
            self.client.follow_user(user_id)
            logger.info(f"Successfully followed user {user_id}")
 
            self.followers_data.append({
                'user_id': user_id,
                'followed_at': datetime.now().isoformat(),
                'followed_back': False
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to follow user {user_id}: {e}")
            return False
    
    def search_tweets(self, query, max_results=100):
        """Search for tweets matching a query with expanded functionality."""
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=["created_at", "author_id", "public_metrics", "context_annotations", "entities"]
            )
            
            if tweets.data:
                logger.info(f"Found {len(tweets.data)} tweets matching the query: '{query}'")
                return tweets.data
            else:
                logger.info(f"No tweets found matching the query: '{query}'")
                return []
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def reply_to_tweet(self, tweet_id, text, media_path=None):
        """Reply to a specific tweet, optionally with media."""
        try:
            media_ids = []
            if media_path:
                media = self.api.media_upload(media_path)
                media_ids = [media.media_id]
            
            reply = self.client.create_tweet(
                text=text,
                in_reply_to_tweet_id=tweet_id,
                media_ids=media_ids if media_ids else None
            )
            
            logger.info(f"Successfully replied to tweet {tweet_id}")
            return reply.data['id']
        except Exception as e:
            logger.error(f"Failed to reply to tweet {tweet_id}: {e}")
            return None
    
    def monitor_mentions(self, since_id=None):
        """Monitor and return new mentions of the authenticated user."""
        try:
            mentions = self.client.get_users_mentions(
                id=self.client.get_me()[0].id,
                since_id=since_id,
                tweet_fields=["created_at", "author_id", "text", "public_metrics"]
            )
            
            if mentions.data:
                logger.info(f"Found {len(mentions.data)} new mentions")
                
           
                mention_ids = [mention.id for mention in mentions.data]
                self.last_mention_id = max(mention_ids)
                
                return mentions.data
            else:
                logger.info("No new mentions found")
                return []
        except Exception as e:
            logger.error(f"Error monitoring mentions: {e}")
            return []
    
  
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of a text using TextBlob."""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
    
            polarity = sentiment.polarity
            
            if polarity > 0.1:
                sentiment_label = "positive"
            elif polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
    
            confidence = abs(polarity)
            
            return {
                'label': sentiment_label,
                'score': confidence,
                'polarity': polarity,
                'subjectivity': sentiment.subjectivity
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'label': 'neutral',
                'score': 0,
                'polarity': 0,
                'subjectivity': 0
            }
    
    def deep_sentiment_analysis(self, text):
        """Perform deeper sentiment analysis using a pre-trained model."""
        try:
            result = self.sentiment_classifier(text)[0]
            return result
        except Exception as e:
            logger.error(f"Error in deep sentiment analysis: {e}")
            return {'label': 'neutral', 'score': 0.5}
    
    def get_trending_topics(self, woeid=1):
        """Get current trending topics (1 is worldwide)."""
        try:
            trends = self.api.get_place_trends(woeid)
            return trends[0]["trends"]
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []
    
    def analyze_user_engagement(self, user_id):
        """Analyze a user's engagement patterns."""
        try:
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=100,
                tweet_fields=["created_at", "public_metrics"]
            )
            
            if not tweets.data:
                return None
            
            engagement_data = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics
                engagement_data.append({
                    'tweet_id': tweet.id,
                    'created_at': tweet.created_at,
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'quotes': metrics.get('quote_count', 0),
                    'total_engagement': (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) +
                        metrics.get('reply_count', 0) +
                        metrics.get('quote_count', 0)
                    )
                })
            
            df = pd.DataFrame(engagement_data)
            
        
            df['hour'] = pd.to_datetime(df['created_at']).dt.hour
            posting_time_distribution = df.groupby('hour').size().to_dict()
            
            hour_engagement = df.groupby('hour')['total_engagement'].mean().to_dict()
            
            optimal_hours = sorted(hour_engagement.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'avg_likes': df['likes'].mean(),
                'avg_retweets': df['retweets'].mean(),
                'avg_replies': df['replies'].mean(),
                'avg_quotes': df['quotes'].mean(),
                'avg_total_engagement': df['total_engagement'].mean(),
                'posting_time_distribution': posting_time_distribution,
                'optimal_posting_hours': optimal_hours
            }
        except Exception as e:
            logger.error(f"Error analyzing user engagement: {e}")
            return None
    
    def generate_analytics_report(self):
        """Generate a comprehensive analytics report for the bot account."""
        try:
            user = self.client.get_me()[0]
            user_id = user.id
            
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=100,
                tweet_fields=["created_at", "public_metrics"]
            )
            
            if not tweets.data:
                return "No tweets found for analysis."
            
            tweet_data = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics
                tweet_data.append({
                    'tweet_id': tweet.id,
                    'created_at': tweet.created_at,
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'quotes': metrics.get('quote_count', 0),
                    'total_engagement': (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) +
                        metrics.get('reply_count', 0) +
                        metrics.get('quote_count', 0)
                    )
                })
            
            df = pd.DataFrame(tweet_data)
            
            report = {
                'period': f"{df['created_at'].min()} to {df['created_at'].max()}",
                'total_tweets': len(df),
                'avg_engagement_per_tweet': df['total_engagement'].mean(),
                'most_liked_tweet_id': df.loc[df['likes'].idxmax()]['tweet_id'] if not df['likes'].empty else None,
                'most_retweeted_tweet_id': df.loc[df['retweets'].idxmax()]['tweet_id'] if not df['retweets'].empty else None,
                'engagement_trend': df.sort_values('created_at')['total_engagement'].tolist(),
                'best_day_of_week': df.assign(day=pd.to_datetime(df['created_at']).dt.day_name()).groupby('day')['total_engagement'].mean().idxmax()
            }
            
            report_path = os.path.join(DATA_FOLDER, "analytics", f"report_{datetime.now().strftime('%Y%m%d')}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f)
            
            return report
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return None
    
    
    def generate_tweet_text(self, prompt, max_length=280):
        """Generate tweet text using GPT-2."""
        try:
            seed_text = prompt.strip()
            if len(seed_text) > 100:
                seed_text = seed_text[:100] 
            
        
            generated_text = self.text_generator(
                seed_text, 
                max_length=max_length, 
                num_return_sequences=3,
                temperature=0.8
            )
            
         
            candidates = []
            for text in generated_text:
                processed_text = text['generated_text'].replace(seed_text, '')
                
                processed_text = processed_text.strip()
                processed_text = re.sub(r'\s+', ' ', processed_text)  
                
                if len(seed_text + processed_text) > 280:
                    processed_text = processed_text[:280 - len(seed_text) - 3] + "..."
                
                full_text = (seed_text + " " + processed_text).strip()
                if full_text and len(full_text) <= 280:
                    candidates.append(full_text)
            
            
            if candidates:
                return sorted(candidates, key=len, reverse=True)[0]
            return seed_text
        except Exception as e:
            logger.error(f"Error generating tweet text: {e}")
            return prompt
    
    def generate_hashtags(self, text, max_tags=3):
        """Generate relevant hashtags for a given text."""
        try:
            
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            
           
            stop_words = set(stopwords.words('english'))
            words = [word for word in clean_text.split() if word not in stop_words and len(word) > 3]
            
           
            word_counts = Counter(words)
            
       
            common_words = word_counts.most_common(max_tags)
            
        
            hashtags = ['#' + word for word, _ in common_words]
            
            return hashtags
        except Exception as e:
            logger.error(f"Error generating hashtags: {e}")
            return []
    
    def create_wordcloud_from_tweets(self, query, filename='wordcloud.png'):
        """Create a word cloud image from tweets matching a query."""
        try:
            tweets = self.search_tweets(query, max_results=100)
            if not tweets:
                return None
            
      
            all_text = ' '.join([tweet.text for tweet in tweets])
            

            clean_text = re.sub(r'http\S+|@\S+|RT|\n', '', all_text)
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            
          
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_text)
            
      
            output_path = os.path.join(DATA_FOLDER, "images", filename)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Word cloud created and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            return None
    
    def create_engagement_graph(self, filename='engagement.png'):
        """Create a graph showing tweet engagement over time."""
        try:
            user_id = self.client.get_me()[0].id
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=100,
                tweet_fields=["created_at", "public_metrics"]
            )
            
            if not tweets.data:
                return None
            
            dates = []
            engagement = []
            
            for tweet in tweets.data:
                metrics = tweet.public_metrics
                total_engagement = (
                    metrics.get('like_count', 0) +
                    metrics.get('retweet_count', 0) +
                    metrics.get('reply_count', 0) +
                    metrics.get('quote_count', 0)
                )
                
                dates.append(tweet.created_at)
                engagement.append(total_engagement)
            
            df = pd.DataFrame({'date': dates, 'engagement': engagement})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['date'], df['engagement'], marker='o', linestyle='-', color='#1DA1F2')
            plt.title('Tweet Engagement Over Time')
            plt.xlabel('Date')
            plt.ylabel('Total Engagement')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_path = os.path.join(DATA_FOLDER, "images", filename)
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Engagement graph created and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating engagement graph: {e}")
            return None
    
    def create_quote_image(self, quote, author=None, filename='quote.png'):
        """Create a shareable quote image."""
        try:
            width, height = 800, 400
            background_color = (240, 240, 240)
            text_color = (20, 20, 20)
            
            image = Image.new('RGB', (width, height), background_color)
            draw = ImageDraw.Draw(image)
            
            try:
                title_font = ImageFont.truetype("arial.ttf", 32)
                author_font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                title_font = ImageFont.load_default()
                author_font = ImageFont.load_default()
            
            max_width = width - 100
            lines = []
            words = quote.split()
            current_line = words[0]
            
            for word in words[1:]:
                test_line = current_line + " " + word
                w, h = draw.textsize(test_line, font=title_font)
                if w <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            lines.append(current_line)
            
            line_height = title_font.getsize("hg")[1] + 10
            text_height = len(lines) * line_height
            
            y_text = (height - text_height) // 2
            
            for line in lines:
                w, h = draw.textsize(line, font=title_font)
                x = (width - w) // 2
                draw.text((x, y_text), line, font=title_font, fill=text_color)
                y_text += line_height
            
            if author:
                author_text = f"— {author}"
                w, h = draw.textsize(author_text, font=author_font)
                x = (width - w) // 2
                draw.text((x, y_text + 20), author_text, font=author_font, fill=text_color)
            
            output_path = os.path.join(DATA_FOLDER, "images", filename)
            image.save(output_path)
            
            logger.info(f"Quote image created and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating quote image: {e}")
            return None
    
    
    def schedule_tweet(self, text, scheduled_time, media_path=None):
        """Schedule a tweet for future posting."""
        tweet_data = {
            'text': text,
            'scheduled_time': scheduled_time.isoformat() if isinstance(scheduled_time, datetime) else scheduled_time,
            'media_path': media_path,
            'id': f"scheduled_{len(self.scheduled_tweets) + 1}"
        }
        
        self.scheduled_tweets.append(tweet_data)
        self._save_data()
        
        logger.info(f"Tweet scheduled for {scheduled_time}")
        return tweet_data['id']
    
    def unschedule_tweet(self, tweet_id):
        """Remove a scheduled tweet."""
        initial_count = len(self.scheduled_tweets)
        self.scheduled_tweets = [t for t in self.scheduled_tweets if t['id'] != tweet_id]
        
        if len(self.scheduled_tweets) < initial_count:
            self._save_data()
            logger.info(f"Tweet {tweet_id} unscheduled")
            return True
        
        logger.warning(f"Tweet {tweet_id} not found in schedule")
        return False
    
    def process_scheduled_tweets(self):
        """Process tweets scheduled for posting."""
        current_time = datetime.now()
        tweets_to_post = []
        
        for tweet in self.scheduled_tweets:
            scheduled_time = datetime.fromisoformat(tweet['scheduled_time'])
            if scheduled_time <= current_time:
                tweets_to_post.append(tweet)
        
        for tweet in tweets_to_post:
            logger.info(f"Posting scheduled tweet: {tweet['id']}")
            self.post_tweet(tweet['text'], tweet['media_path'])
            self.scheduled_tweets.remove(tweet)
        
        if tweets_to_post:
            self._save_data()
    
    def add_topic_to_monitor(self, topic, sentiment_threshold=0.0, engagement_threshold=10):
        """Add a topic to monitor for interesting tweets."""
        topic_data = {
            'query': topic,
            'sentiment_threshold': sentiment_threshold,
            'engagement_threshold': engagement_threshold,
            'last_checked': datetime.now().isoformat(),
            'id': f"topic_{len(self.topics_to_monitor) + 1}"
        }
        
        self.topics_to_monitor.append(topic_data)
        self._save_data()
        
        logger.info(f"Added topic to monitor: {topic}")
        return topic_data['id']
    
    def remove_monitored_topic(self, topic_id):
        """Remove a topic from monitoring."""
        initial_count = len(self.topics_to_monitor)
        self.topics_to_monitor = [t for t in self.topics_to_monitor if t['id'] != topic_id]
        
        if len(self.topics_to_monitor) < initial_count:
            self._save_data()
            logger.info(f"Topic {topic_id} removed from monitoring")
            return True
        
        logger.warning(f"Topic {topic_id} not found")
        return False
    
    def monitor_topics(self):
        """Check monitored topics for interesting tweets."""
        interesting_tweets = []
        
        for topic in self.topics_to_monitor:
            logger.info(f"Checking topic: {topic['query']}")
            
            topic['last_checked'] = datetime.now().isoformat()
            
            tweets = self.search_tweets(topic['query'], max_results=30)
            
            for tweet in tweets:
                metrics = tweet.public_metrics
                engagement = metrics.get('like_count', 0) + metrics.get('retweet_count', 0)
                
                if engagement < topic['engagement_threshold']:
                    continue
                
                sentiment = self.analyze_sentiment(tweet.text)
                
                if abs(sentiment['polarity']) >= topic['sentiment_threshold']:
                    interesting_tweets.append({
                        'tweet_id': tweet.id,
                        'author_id': tweet.author_id,
                        'text': tweet.text,
                        'engagement': engagement,
                        'sentiment': sentiment,
                        'topic': topic['query']
                    })
        
        self._save_data()
        
        return interesting_tweets
    
    def detect_optimal_posting_times(self, lookback_days=30):
        """Analyze past engagement to detect optimal posting times."""
        try:
            user_id = self.client.get_me()[0].id
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=100,
                tweet_fields=["created_at", "public_metrics"],
                start_time=datetime.now() - timedelta(days=lookback_days)
            )
            
            if not tweets.data:
                logger.info("No tweets found for analysis")
                return None
            
            engagement_data = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics
                engagement_data.append({
                    'created_at': tweet.created_at,
                    'hour': tweet.created_at.hour,
                    'day_of_week': tweet.created_at.strftime('%A'),
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'total_engagement': (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) +
                        metrics.get('reply_count', 0)
                    )
                })
            
            df = pd.DataFrame(engagement_data)
            
            hour_engagement = df.groupby('hour')['total_engagement'].mean().sort_values(ascending=False)
            
            day_engagement = df.groupby('day_of_week')['total_engagement'].mean()
            
            best_hours = hour_engagement.head(3).index.tolist()
            
            best_day = day_engagement.idxmax()
            
            results = {
                'best_hours': best_hours,
                'best_day': best_day,
                'hourly_engagement': hour_engagement.to_dict(),
                'daily_engagement': day_engagement.to_dict(),
                'average_engagement': df['total_engagement'].mean(),
                'tweets_analyzed': len(df)
            }
            
            logger.info(f"Optimal posting times analysis complete. Best hours: {best_hours}, Best day: {best_day}")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting optimal posting times: {e}")
            return None


if __name__ == "__main__":
    bot = AdvancedTwitterBot()
