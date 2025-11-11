"""
Altmetric API Data Extractor
Extracts complete Altmetric data including policy mentions, news, blogs, etc.
Outputs to Excel with all available metrics
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Any
import json
from openpyxl.utils import get_column_letter

class AltmetricDataExtractor:
    """Extract comprehensive data from Altmetric API"""
    
    def __init__(self, api_url: str):
        """
        Initialize with your Altmetric API URL
        
        Args:
            api_url: The API URL from Altmetric Explorer's "Open results in API"
        """
        self.api_url = api_url
        self.all_results = []
        
    def fetch_all_pages(self) -> List[Dict]:
        """
        Fetch all pages of results from the API
        
        Returns:
            List of all research output dictionaries
        """
        current_url = self.api_url
        page = 1
        
        while current_url:
            print(f"Fetching page {page}...")
            
            try:
                response = requests.get(current_url)
                response.raise_for_status()
                data = response.json()
                
                # Extract research outputs from this page
                if 'data' in data:
                    self.all_results.extend(data['data'])
                    print(f"  - Retrieved {len(data['data'])} records")
                
                # Check for next page
                if 'links' in data and 'next' in data['links']:
                    current_url = data['links']['next']
                    page += 1
                    time.sleep(1)  # Be polite to the API
                else:
                    current_url = None
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                break
                
        print(f"\nTotal records retrieved: {len(self.all_results)}")
        return self.all_results
    
    def extract_policy_data(self, attributes: Dict) -> Dict:
        """
        Extract policy-related metrics
        
        Note: The list endpoint only provides counts. Detailed policy_sources 
        (names, dates, URLs) are only available when fetching individual articles.
        To get full policy details, you would need to fetch each article separately.
        """
        mentions = attributes.get('mentions', {})
        
        return {
            'policy_mentions_total': mentions.get('policy', 0)
            # Note: policy_sources not available in list endpoint
            # Would require individual article API calls
        }
    
    def extract_news_data(self, attributes: Dict) -> Dict:
        """Extract news and media metrics"""
        mentions = attributes.get('mentions', {})
        
        return {
            'news_msm_total': mentions.get('msm', 0),  # mainstream media
            'blogs_total': mentions.get('blog', 0),
            'reddit_total': mentions.get('rdt', 0),
            'twitter_total': mentions.get('tweet', 0),
            'facebook_total': mentions.get('fbwall', 0),
            'bluesky_total': mentions.get('bluesky', 0),
            'wikipedia_mentions': mentions.get('wikipedia', 0),
            'video_mentions': mentions.get('video', 0),
            'podcast_mentions': mentions.get('podcast', 0),
            'qna_mentions': mentions.get('qna', 0),
            'guideline_mentions': mentions.get('guideline', 0)
        }
    
    def extract_academic_data(self, attributes: Dict) -> Dict:
        """Extract academic engagement metrics"""
        mentions = attributes.get('mentions', {})
        
        return {
            'readers_mendeley': attributes.get('readers', {}).get('mendeley', 0),
            'citations': attributes.get('dimensions', {}).get('citations', 0),
            'peer_reviews': mentions.get('peer_review', 0),
            'patent_mentions': mentions.get('patent', 0),
            'f1000_mentions': mentions.get('f1000', 0)
        }
    
    def extract_basic_info(self, item: Dict) -> Dict:
        """Extract basic article information - NO TRUNCATION"""
        attributes = item.get('attributes', {})
        identifiers = attributes.get('identifiers', {})
        
        return {
            'altmetric_id': item.get('id'),
            'title': attributes.get('title', ''),  # FULL TITLE - no truncation
            'doi': identifiers.get('dois', [''])[0] if identifiers.get('dois') else '',
            'pubmed_id': '; '.join(identifiers.get('pubmed-ids', [])),
            'pmc_id': identifiers.get('pmc-ids', [''])[0] if identifiers.get('pmc-ids') else '',
            'publication_date': attributes.get('publication-date', ''),
            'journal': attributes.get('journal', ''),
            'altmetric_score': attributes.get('altmetric-score', 0),
            'output_type': attributes.get('output-type', ''),
            'oa_status': attributes.get('oa-status', ''),
            'oa_type': attributes.get('oa-type', '')
        }
    
    def create_dataset(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame from all results
        
        Returns:
            DataFrame with all extracted metrics
        """
        if not self.all_results:
            print("No results to process. Run fetch_all_pages() first.")
            return pd.DataFrame()
        
        print("\nProcessing results...")
        processed_data = []
        
        for item in self.all_results:
            attributes = item.get('attributes', {})
            
            # Combine all extracted data
            record = {}
            record.update(self.extract_basic_info(item))
            record.update(self.extract_policy_data(attributes))
            record.update(self.extract_news_data(attributes))
            record.update(self.extract_academic_data(attributes))
            
            processed_data.append(record)
        
        df = pd.DataFrame(processed_data)
        print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def save_to_excel(self, df: pd.DataFrame, output_path: str):
        """
        Save DataFrame to Excel with proper formatting
        - Full titles visible (no truncation)
        - Proper column widths
        - Works with columns beyond Z
        
        Args:
            df: DataFrame to save
            output_path: Path for output Excel file
        """
        print(f"\nSaving to {output_path}...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Altmetric Data', index=False)
            
            # Access the worksheet
            worksheet = writer.sheets['Altmetric Data']
            
            # Auto-adjust column widths with special handling for title column
            for idx, col in enumerate(df.columns):
                # Calculate maximum length needed
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                
                # Special handling for title column - allow up to 100 characters wide
                if col.lower() == 'title':
                    column_width = min(max_length + 2, 100)
                else:
                    column_width = min(max_length + 2, 50)
                
                # Use proper Excel column letter (works for columns beyond Z)
                column_letter = get_column_letter(idx + 1)
                worksheet.column_dimensions[column_letter].width = column_width
        
        print(f"✓ Successfully saved {len(df)} records to {output_path}")


def main():
    """
    Main execution function
    Usage example
    """
    # REPLACE THIS with your actual API URL from Altmetric
    API_URL = "https://www.altmetric.com/explorer/api/research_outputs?digest=1213441e039fc8fb4bb90f80a4061a296b1ac7f8&filter%5Bidentifier_list_id%5D=ad1030f9-915b-494e-a07d-6890d69d979a&filter%5Bscope%5D=all&filter%5Bview%5D=list&key=fb1d3a1ba457433cbb7e1de999e1033c"
    
    # Example: 
    # API_URL = "https://www.altmetric.com/explorer/api/research_outputs?digest=abc123&filter[identifier_list_id]=ad1030f9-915b-494e-a07d-6890d69d979a"
    
    if API_URL == "" or API_URL == "YOUR_API_URL_HERE":
        print("="*60)
        print("ERROR: API URL NOT SET")
        print("="*60)
        print("\nPlease replace API_URL with your actual Altmetric API URL")
        print("\nTo get your API URL:")
        print("1. In Altmetric Explorer, after your search, click 'EXPORT THIS TAB'")
        print("2. Select 'Open results in API'")
        print("3. Copy the URL from the browser")
        print("4. Paste it in this script where it says API_URL = ...")
        print("\nExample URL format:")
        print("https://www.altmetric.com/explorer/api/research_outputs?digest=...")
        return
    
    # Initialize extractor
    print("="*60)
    print("ALTMETRIC API DATA EXTRACTOR")
    print("="*60)
    print(f"\nAPI URL: {API_URL[:100]}...\n")
    
    extractor = AltmetricDataExtractor(API_URL)
    
    # Fetch all data
    extractor.fetch_all_pages()
    
    if not extractor.all_results:
        print("\n❌ No data retrieved. Please check your API URL.")
        return
    
    # Create dataset
    df = extractor.create_dataset()
    
    # Save to Excel - SAME PATH AS TEST FILE
    output_path = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\03_Almetrics_API_CS.xlsx"
    extractor.save_to_excel(df, output_path)
    
    # Display summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total publications: {len(df)}")
    print(f"\nPublications with policy mentions: {(df['policy_mentions_total'] > 0).sum()}")
    print(f"Publications with news coverage: {(df['news_msm_total'] > 0).sum()}")
    print(f"Publications with citations: {(df['citations'] > 0).sum()}")
    print(f"\nAverage Altmetric score: {df['altmetric_score'].mean():.2f}")
    print(f"Median policy mentions: {df['policy_mentions_total'].median():.0f}")
    
    # Show title length statistics
    df['title_length'] = df['title'].str.len()
    print(f"\nTitle length statistics:")
    print(f"  - Shortest title: {df['title_length'].min()} characters")
    print(f"  - Longest title: {df['title_length'].max()} characters")
    print(f"  - Average title: {df['title_length'].mean():.0f} characters")
    
    print("\n" + "="*60)
    print("✓ EXTRACTION COMPLETE")
    print("="*60)
    print(f"\nYour complete dataset is ready at:")
    print(f"{output_path}")
    print("\nThe title column contains FULL titles (no truncation)")


if __name__ == "__main__":
    main()