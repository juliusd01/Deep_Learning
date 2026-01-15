"""
Scraper using the WordPress JSON API endpoint from rollcall.com
Successfully extracts Trump's Truth Social posts about specific topics
"""

import requests
import json
from datetime import datetime
from urllib.parse import urlencode


def scrape_via_api(search_query='ukraine', platform='truth social', page=1, per_page=50):
    """
    Scrape posts using the WordPress JSON API
    
    Args:
        search_query: Search term
        platform: Platform filter
        page: Page number
        per_page: Results per page
    
    Returns:
        Dictionary with posts and metadata
    """
    
    base_url = "https://rollcall.com"
    api_endpoint = "/wp-json/factbase/v1/twitter"
    
    # Build query parameters
    params = {
        'q': search_query,
        'platform': platform,
        'sort': 'date',
        'sort_order': 'desc',
        'page': page,
        'per_page': per_page
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://rollcall.com/factbase/trump/topic/social/'
    }
    
    url = f"{base_url}{api_endpoint}?{urlencode(params)}"
    
    print(f"Requesting API: {url}\n")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Response Length: {len(response.text)} chars\n")
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Save raw response
                with open('api_raw_response.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print("✓ Saved raw API response to api_raw_response.json\n")
                
                return {
                    'success': True,
                    'data': data,
                    'query_params': params,
                    'fetched_at': datetime.now().isoformat()
                }
                
            except json.JSONDecodeError as e:
                print(f"✗ Failed to parse JSON: {e}")
                print(f"Response preview: {response.text[:500]}")
                return {'success': False, 'error': 'Invalid JSON response'}
        else:
            print(f"✗ API returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return {'success': False, 'error': f'HTTP {response.status_code}'}
            
    except requests.RequestException as e:
        print(f"✗ Request error: {e}")
        return {'success': False, 'error': str(e)}


def parse_posts(api_data):
    """
    Parse posts from API response
    
    Args:
        api_data: Raw API response data
    
    Returns:
        List of parsed posts
    """
    posts = []
    
    # The API response structure may vary, try different approaches
    data = api_data.get('data', api_data)
    
    # Try to find posts in various possible structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (
            data.get('posts') or 
            data.get('items') or 
            data.get('results') or 
            data.get('statements') or
            data.get('data', [])
        )
    else:
        items = []
    
    for item in items:
        if isinstance(item, dict):
            post = {
                'id': item.get('id'),
                'date': item.get('date') or item.get('created_at') or item.get('published_at'),
                'content': item.get('content') or item.get('text') or item.get('message') or item.get('body'),
                'platform': item.get('platform') or item.get('source'),
                'url': item.get('url') or item.get('link') or item.get('permalink'),
                'author': item.get('author') or item.get('user'),
                'metadata': {k: v for k, v in item.items() if k not in ['id', 'date', 'content', 'platform', 'url', 'author']}
            }
            posts.append(post)
    
    return posts


def scrape_multiple_pages(search_query='ukraine', platform='truth social', max_pages=5):
    """Scrape multiple pages of results"""
    
    all_posts = []
    
    print("="*80)
    print("MULTI-PAGE API SCRAPER")
    print("="*80)
    print(f"Search: {search_query}")
    print(f"Platform: {platform}")
    print(f"Max pages: {max_pages}")
    print("="*80)
    print()
    
    for page in range(1, max_pages + 1):
        print(f"--- Page {page} ---")
        
        result = scrape_via_api(search_query, platform, page=page, per_page=50)
        
        if result['success']:
            posts = parse_posts(result['data'])
            all_posts.extend(posts)
            
            print(f"✓ Extracted {len(posts)} posts from page {page}")
            print(f"Total posts so far: {len(all_posts)}\n")
            
            # If we got no posts, we've reached the end
            if not posts:
                print("No more posts found. Stopping.\n")
                break
        else:
            print(f"✗ Failed to fetch page {page}: {result.get('error')}\n")
            break
        
        # Be polite, wait between requests
        if page < max_pages:
            import time
            time.sleep(1)
    
    return all_posts


def display_posts(posts, limit=5):
    """Display posts in a readable format"""
    
    print("\n" + "="*80)
    print(f"SCRAPED POSTS (showing {min(limit, len(posts))} of {len(posts)})")
    print("="*80)
    
    for i, post in enumerate(posts[:limit], 1):
        print(f"\n{'─'*80}")
        print(f"POST #{i}")
        print(f"{'─'*80}")
        print(f"Date: {post.get('date', 'N/A')}")
        print(f"Platform: {post.get('platform', 'N/A')}")
        if post.get('url'):
            print(f"URL: {post['url']}")
        print(f"\nContent:")
        content = post.get('content', 'N/A')
        if content and len(content) > 300:
            print(content[:300] + "...")
        else:
            print(content)
        
        if post.get('metadata'):
            print(f"\nMetadata keys: {list(post['metadata'].keys())}")


def save_posts_to_file(posts, filename='rollcall_posts_final.json'):
    """Save posts to JSON file"""
    
    output = {
        'total_posts': len(posts),
        'scraped_at': datetime.now().isoformat(),
        'source': 'rollcall.com via WordPress API',
        'posts': posts
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(posts)} posts to {filename}")
    
    # Also save as CSV for easy viewing
    try:
        import csv
        csv_filename = filename.replace('.json', '.csv')
        
        if posts:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                # Get all possible field names
                fieldnames = set()
                for post in posts:
                    fieldnames.update(post.keys())
                    if 'metadata' in post and isinstance(post['metadata'], dict):
                        fieldnames.update([f"meta_{k}" for k in post['metadata'].keys()])
                fieldnames.discard('metadata')
                fieldnames = sorted(list(fieldnames))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for post in posts:
                    row = {k: v for k, v in post.items() if k != 'metadata'}
                    # Flatten metadata
                    if 'metadata' in post and isinstance(post['metadata'], dict):
                        for k, v in post['metadata'].items():
                            row[f'meta_{k}'] = v
                    writer.writerow(row)
            
            print(f"✓ Saved posts to CSV: {csv_filename}")
    except Exception as e:
        print(f"Note: Could not save CSV: {e}")


def main():
    """Main scraping workflow"""
    
    # Scrape posts
    posts = scrape_multiple_pages(
        search_query='ukraine',
        platform='truth social',
        max_pages=10  # Adjust as needed
    )
    
    if posts:
        # Display sample
        display_posts(posts, limit=5)
        
        # Save to file
        save_posts_to_file(posts)
        
        print("\n" + "="*80)
        print("SCRAPING COMPLETE!")
        print("="*80)
        print(f"Total posts scraped: {len(posts)}")
        print("\nFiles created:")
        print("  - rollcall_posts_final.json (full data)")
        print("  - rollcall_posts_final.csv (spreadsheet format)")
        print("  - api_raw_response.json (raw API response)")
        
    else:
        print("\n" + "="*80)
        print("NO POSTS FOUND")
        print("="*80)
        print("\nThis could mean:")
        print("  1. The API endpoint or parameters are different than expected")
        print("  2. The search returned no results")
        print("  3. The API requires authentication")
        print("\nCheck api_raw_response.json if it was created to see the actual response.")


if __name__ == "__main__":
    main()
