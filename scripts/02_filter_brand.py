import pandas as pd
import numpy as np

print("Loading dataset... (may take 60 seconds)")
df = pd.read_csv('data/twcs/twcs.csv')
print(f"Total rows: {len(df)}")

BRAND_AUTHOR_ID = 'AmazonHelp'

print(f"\nFiltering for brand: {BRAND_AUTHOR_ID}")

# Index all tweets by tweet_id for fast lookup
print("Building tweet index...")
tweet_index = df.set_index('tweet_id').to_dict('index')

# Get all brand responses
brand_responses = df[df['author_id'] == BRAND_AUTHOR_ID]
print(f"Amazon response count: {len(brand_responses)}")

# Build conversation pairs
print("Building conversation pairs...")
conversations = []

for _, brand_tweet in brand_responses.iterrows():
    reply_to_id = brand_tweet.get('in_response_to_tweet_id')
    
    if pd.isna(reply_to_id):
        continue
    
    reply_to_id = int(reply_to_id)
    
    # Get the customer message
    customer_tweet = tweet_index.get(reply_to_id)
    if customer_tweet is None:
        continue
    
    # Only include inbound (customer) messages
    if not customer_tweet.get('inbound', True):
        continue
    
    conversations.append({
        'conversation_id': brand_tweet.get('conversation_id', ''),
        'customer_tweet_id': reply_to_id,
        'customer_message': str(customer_tweet['text']),
        'customer_author': str(customer_tweet.get('author_id', '')),
        'brand_tweet_id': brand_tweet['tweet_id'],
        'brand_reply': str(brand_tweet['text']),
    })

conv_df = pd.DataFrame(conversations)
print(f"\nTotal conversation pairs built: {len(conv_df)}")

print("\nSample conversations:")
for i, row in conv_df.head(5).iterrows():
    print(f"\n--- Conv {i+1} ---")
    print(f"CUSTOMER: {row['customer_message'][:200]}")
    print(f"AMAZON:   {row['brand_reply'][:200]}")

# Save full
conv_df.to_csv('data/brand_conversations.csv', index=False)
print(f"\nSaved full dataset: data/brand_conversations.csv")

# Save sample of 5000 for working
sample_size = min(5000, len(conv_df))
sample_df = conv_df.sample(n=sample_size, random_state=42)
sample_df.to_csv('data/brand_conversations_sample.csv', index=False)
print(f"Saved sample: data/brand_conversations_sample.csv ({sample_size} rows)")

# Show some stats
print(f"\n--- Quick Stats ---")
print(f"Avg customer message length: {conv_df['customer_message'].str.len().mean():.0f} chars")
print(f"Avg Amazon reply length: {conv_df['brand_reply'].str.len().mean():.0f} chars")
print(f"Sample customer messages:")
for msg in conv_df['customer_message'].sample(5, random_state=1).tolist():
    print(f"  - {msg[:120]}")