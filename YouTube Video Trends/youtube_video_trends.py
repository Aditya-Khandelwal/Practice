#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

#for interacting with the YouTube Api
from googleapiclient.discovery import build
import pandas as pd #for handling and saving data in csv format
import time #For adding delays to avoid hitting API rate limits
from datetime import datetime  #for handling date and time


# In[2]:


# Intialize Youtube API

API_KEY= 'AIzaSyB1fuxX68FeRHAHrEhde3KJQNWK86QmYOw'

#Initialize Youtube API
youtube=build('youtube','v3',developerKey=API_KEY)

# List of regions
# List of regions
regions = ["US", "GB", "CA", "IN", "AU", "DE", "FR", "IT", "ES", "MX", "BR", "JP", "KR", "RU",
           "AR", "CO", "CL", "PE", "VE", "ZA", "NG", "PH", "TH", "ID", "MY", "SG", "HK", "TW",
           "PL", "NL", "BE", "AT", "CH", "SE", "NO", "DK", "FI", "IE", "HU", "CZ", "RO", "GR",
           "TR", "SA", "AE", "KW", "QA", "OM", "JO", "BD", "LK", "MM", "KH", "LA", "NP", "TW",
           "SG", "MD", "BY", "UA", "LT", "LV", "EE", "RS", "HR", "SI", "BG", "MK", "AL", "MT",
           "IS", "GL", "SM", "AD", "MC", "VA", "LI", "AW", "BQ", "CW", "SX"]


# In[3]:


# Function to fetch trending videos
def fetch_trending_videos(region_code='US', max_results=50, page_token=None):
    try:
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails,status',
            chart='mostPopular',
            regionCode=region_code,
            maxResults=max_results,
            pageToken=page_token
        )
        response = request.execute()
        return response
    except Exception as e:
        print(f"Error fetching data for region {region_code}: {e}")
        return None


# In[4]:


# Function to save videos to a single CSV
def save_videos_to_csv(videos):
    df = pd.DataFrame(videos)
    filename = 'youtube_trending_videos_all_regions.csv'
    df.to_csv(filename, index=False)
    print(f"Saved {len(videos)} records to '{filename}'")


# In[5]:


# Main Function
def main():
    all_videos = []
    max_results = 50  # Fetch 50 results per request

    for region in regions:
        next_page_token = None

        while len(all_videos) < 10000:  # Adjust number of results as needed
            print(f"Fetching page with token: {next_page_token} for region {region}")
            response = fetch_trending_videos(region_code=region, max_results=max_results, page_token=next_page_token)

            if response is None:
                print(f"Failed to fetch data for region {region}. Exiting.")
                break

            for video in response.get('items', []):
                video_data = {
                    'video_id': video['id'],
                    'region': region,
                    'trending_date': datetime.now().strftime('%Y-%m-%d'),
                    'title': video['snippet']['title'],
                    'channel_title': video['snippet']['channelTitle'],
                    'category_id': video['snippet']['categoryId'],
                    'publish_time': video['snippet']['publishedAt'],
                    'tags': ','.join(video['snippet'].get('tags', [])),
                    'views': int(video['statistics'].get('viewCount', 0)),
                    'likes': int(video['statistics'].get('likeCount', 0)),
                    'dislikes': int(video['statistics'].get('dislikeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0)),
                    'thumbnail_link': video['snippet']['thumbnails']['high']['url'],
                    'comments_disabled': 'commentCount' not in video['statistics'],
                    'ratings_disabled': 'likeCount' not in video['statistics'] or 'dislikeCount' not in video['statistics'],
                    'video_error_or_removed': video['status'].get('uploadStatus') != 'processed',
                    'description': video['snippet']['description']
                }
                all_videos.append(video_data)

            next_page_token = response.get('nextPageToken')

            if not next_page_token or len(all_videos) >= 10000:  # Adjust the number of records as needed
                break

            time.sleep(1)  # Avoid hitting rate limits

    save_videos_to_csv(all_videos)
    print(f"Completed fetching data for all regions. Total records: {len(all_videos)}")

# Run the Main Function
if __name__ == '__main__':
    main()


# In[ ]:




