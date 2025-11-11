"""
Quick Test Version - Altmetric Data Extractor
Tests with just the first page (100 records) to verify everything works
"""

import requests
import pandas as pd
import json

def test_altmetric_api(api_url: str):
    """
    Quick test with first page only
    
    Args:
        api_url: Your API URL from Altmetric Explorer
    """
    print("Fetching first page of results...")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data:
            print("Error: No data found in API response")
            return None
        
        print(f"✓ Successfully retrieved {len(data['data'])} records\n")
        
        # Show structure of first record
        if data['data']:
            first_record = data['data'][0]
            print("="*60)
            print("SAMPLE RECORD STRUCTURE")
            print("="*60)
            
            attributes = first_record.get('attributes', {})
            identifiers = attributes.get('identifiers', {})
            
            print(f"\nTitle: {attributes.get('title', 'N/A')}")
            print(f"DOI: {identifiers.get('dois', ['N/A'])[0]}")
            print(f"Altmetric Score: {attributes.get('altmetric-score', 0)}")
            
            mentions = attributes.get('mentions', {})
            print(f"\nPolicy mentions: {mentions.get('policy', 0)}")
            print(f"News (MSM) mentions: {mentions.get('msm', 0)}")
            print(f"Twitter mentions: {mentions.get('tweet', 0)}")
            
            # Show policy sources if available
            policy_sources = attributes.get('policy_sources', [])
            if policy_sources:
                print(f"\nPolicy sources found: {len(policy_sources)}")
                print("First policy source:")
                print(f"  - Name: {policy_sources[0].get('name', 'N/A')}")
                print(f"  - Posted: {policy_sources[0].get('posted_on', 'N/A')}")
            else:
                print("\nNo policy sources in this record")
            
        # Create simple DataFrame
        print("\n" + "="*60)
        print("CREATING SAMPLE DATASET")
        print("="*60)
        
        records = []
        for item in data['data']:
            attrs = item.get('attributes', {})
            ids = attrs.get('identifiers', {})
            mentions = attrs.get('mentions', {})
            
            records.append({
                'title': attrs.get('title', '')[:100] + '...',  # Truncate for display
                'doi': ids.get('dois', [''])[0] if ids.get('dois') else '',
                'altmetric_score': attrs.get('altmetric-score', 0),
                'policy_mentions': mentions.get('policy', 0),
                'news_msm': mentions.get('msm', 0),  # mainstream media
                'twitter': mentions.get('tweet', 0),
                'citations': attrs.get('dimensions', {}).get('citations', 0)
            })
        
        df = pd.DataFrame(records)
        
        print(f"\nCreated DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print("\nSummary statistics:")
        print(df.describe())
        
        print(f"\nRecords with policy mentions: {(df['policy_mentions'] > 0).sum()}")
        print(f"Records with news mentions: {(df['news_msm'] > 0).sum()}")
        
        # Save test output - CORRECTED PATH WITH FILENAME
        output_path = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\02_altmetrics_api.xlsx"
        df.to_excel(output_path, index=False) 
        print(f"\n✓ Test sample saved to: {output_path}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API URL is complete and correct")
        print("2. Make sure you copied the ENTIRE URL from the browser")
        print("3. Try generating a new URL from Altmetric Explorer")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def main():
    """Test with your API URL"""
    
    # REPLACE THIS with your actual API URL
    API_URL = ""
    
    if API_URL == "YOUR_API_URL_HERE":
        print("="*60)
        print("SETUP REQUIRED")
        print("="*60)
        print("\nPlease replace API_URL with your actual Altmetric API URL")
        print("\nSteps:")
        print("1. Go to Altmetric Explorer with your search results")
        print("2. Click 'EXPORT THIS TAB'")
        print("3. Click 'Open results in API'")
        print("4. Copy the URL from your browser")
        print("5. Paste it in this file where it says API_URL = ...")
        print("\nExample URL format:")
        print("https://www.altmetric.com/explorer/api/research_outputs?digest=...")
        return
    
    print("="*60)
    print("ALTMETRIC API TEST")
    print("="*60)
    print(f"\nTesting with URL:\n{API_URL[:100]}...\n")
    
    df = test_altmetric_api(API_URL)
    
    if df is not None:
        print("\n" + "="*60)
        print("TEST SUCCESSFUL!")
        print("="*60)
        print("\nYou can now run the full extractor script")
    else:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print("\nPlease check the error messages above and try again")


if __name__ == "__main__":
    main()