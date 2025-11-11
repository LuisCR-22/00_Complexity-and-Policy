"""
OpenAlex API Data Extractor - TEST VERSION
Tests extraction on first 10 papers only
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Any
import json
from openpyxl.utils import get_column_letter

class OpenAlexDataExtractor:
    """Extract comprehensive data from OpenAlex API"""
    
    def __init__(self, input_csv_path: str, test_mode: bool = True, test_size: int = 10):
        """
        Initialize with input CSV path containing OpenAlex IDs
        
        Args:
            input_csv_path: Path to CSV file with Node_ID column
            test_mode: If True, only process first test_size papers
            test_size: Number of papers to process in test mode
        """
        self.input_csv_path = input_csv_path
        self.test_mode = test_mode
        self.test_size = test_size
        self.all_results = []
        
    def transform_id(self, openalex_url: str) -> str:
        """
        Transform OpenAlex URL to API URL
        
        Example: https://openalex.org/W2008620264 → https://api.openalex.org/W2008620264
        """
        return openalex_url.replace('openalex.org', 'api.openalex.org')
    
    def fetch_work_data(self, work_id: str) -> Dict:
        """
        Fetch data for a single work from OpenAlex API
        
        Args:
            work_id: OpenAlex work ID URL
            
        Returns:
            JSON data from API or None if error
        """
        api_url = self.transform_id(work_id)
        
        print(f"  🔗 API URL: {api_url}")
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            print(f"  ✓ Successfully fetched data (Status: {response.status_code})")
            return data
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching {work_id}: {e}")
            return None
    
    def extract_authors_data(self, authorships: List[Dict]) -> Dict:
        """Extract detailed author information"""
        authors_names = []
        all_institutions = []
        all_institution_ids = []
        all_countries = []
        author_affiliations = []
        
        for authorship in authorships:
            # Author name
            author = authorship.get('author', {})
            author_name = author.get('display_name', '')
            if author_name:
                authors_names.append(author_name)
            
            # Institutions per author
            institutions = authorship.get('institutions', [])
            author_insts = []
            for inst in institutions:
                inst_name = inst.get('display_name', '')
                inst_id = inst.get('id', '')
                country = inst.get('country_code', '')
                
                if inst_name:
                    author_insts.append(inst_name)
                    if inst_name not in all_institutions:
                        all_institutions.append(inst_name)
                
                if inst_id and inst_id not in all_institution_ids:
                    all_institution_ids.append(inst_id)
                
                if country and country not in all_countries:
                    all_countries.append(country)
            
            author_affiliations.append('; '.join(author_insts) if author_insts else 'No affiliation')
        
        return {
            'authors_names': ' | '.join(authors_names) if authors_names else '',
            'author_affiliations': ' | '.join(author_affiliations) if author_affiliations else '',
            'institutions': '; '.join(all_institutions) if all_institutions else '',
            'institution_ids': '; '.join(all_institution_ids) if all_institution_ids else '',
            'countries': '; '.join(all_countries) if all_countries else '',
            'authors_count': len(authors_names)
        }
    
    def extract_topics_data(self, data: Dict) -> Dict:
        """Extract comprehensive topic information"""
        result = {}
        
        # Primary topic
        primary_topic = data.get('primary_topic', {})
        if primary_topic:
            result['primary_topic_id'] = primary_topic.get('id', '')
            result['primary_topic'] = primary_topic.get('display_name', '')
            result['primary_topic_score'] = primary_topic.get('score', 0)
            
            subfield = primary_topic.get('subfield', {})
            result['primary_subfield_id'] = subfield.get('id', '') if subfield else ''
            result['primary_subfield'] = subfield.get('display_name', '') if subfield else ''
            
            field = primary_topic.get('field', {})
            result['primary_field_id'] = field.get('id', '') if field else ''
            result['primary_field'] = field.get('display_name', '') if field else ''
            
            domain = primary_topic.get('domain', {})
            result['primary_domain_id'] = domain.get('id', '') if domain else ''
            result['primary_domain'] = domain.get('display_name', '') if domain else ''
        else:
            result['primary_topic_id'] = ''
            result['primary_topic'] = ''
            result['primary_topic_score'] = 0
            for level in ['subfield', 'field', 'domain']:
                result[f'primary_{level}_id'] = ''
                result[f'primary_{level}'] = ''
        
        # All topics (secondary and tertiary)
        topics = data.get('topics', [])
        topic_labels = ['primary', 'secondary', 'tertiary']
        
        for i in range(3):
            prefix = topic_labels[i]
            if i < len(topics):
                topic = topics[i]
                if i > 0:  # Skip primary as already handled
                    result[f'{prefix}_topic_id'] = topic.get('id', '')
                    result[f'{prefix}_topic'] = topic.get('display_name', '')
                    result[f'{prefix}_topic_score'] = topic.get('score', 0)
                    
                    subfield = topic.get('subfield', {})
                    result[f'{prefix}_subfield'] = subfield.get('display_name', '') if subfield else ''
                    result[f'{prefix}_subfield_id'] = subfield.get('id', '') if subfield else ''
                    
                    field = topic.get('field', {})
                    result[f'{prefix}_field'] = field.get('display_name', '') if field else ''
                    result[f'{prefix}_field_id'] = field.get('id', '') if field else ''
                    
                    domain = topic.get('domain', {})
                    result[f'{prefix}_domain'] = domain.get('display_name', '') if domain else ''
                    result[f'{prefix}_domain_id'] = domain.get('id', '') if domain else ''
            else:
                if i > 0:
                    result[f'{prefix}_topic_id'] = ''
                    result[f'{prefix}_topic'] = ''
                    result[f'{prefix}_topic_score'] = 0
                    result[f'{prefix}_subfield'] = ''
                    result[f'{prefix}_subfield_id'] = ''
                    result[f'{prefix}_field'] = ''
                    result[f'{prefix}_field_id'] = ''
                    result[f'{prefix}_domain'] = ''
                    result[f'{prefix}_domain_id'] = ''
        
        return result
    
    def extract_keywords(self, keywords: List[Dict]) -> Dict:
        """Extract up to 6 main keywords"""
        result = {}
        for i in range(6):
            if i < len(keywords):
                result[f'keyword_{i+1}'] = keywords[i].get('display_name', '')
            else:
                result[f'keyword_{i+1}'] = ''
        
        # Also provide concatenated version
        keyword_list = [kw.get('display_name', '') for kw in keywords[:6] if kw.get('display_name')]
        result['keywords_all'] = '; '.join(keyword_list)
        
        return result
    
    def extract_concepts(self, concepts: List[Dict]) -> str:
        """Extract concepts with scores"""
        concept_strings = []
        for concept in concepts[:10]:  # Top 10 concepts
            name = concept.get('display_name', '')
            score = concept.get('score', 0)
            if name:
                concept_strings.append(f"{name} ({score:.3f})")
        
        return '; '.join(concept_strings)
    
    def extract_locations(self, locations: List[Dict]) -> Dict:
        """Extract location information - simplified and robust"""
        result = {'locations_count': len(locations)}
        
        # Get first 4 locations - just source name and first ISSN
        for i in range(4):
            idx = i + 1
            
            if i < len(locations):
                location = locations[i]
                source = location.get('source')
                
                # Source name
                if source and isinstance(source, dict):
                    result[f'location_{idx}_source'] = source.get('display_name') or ''
                    
                    # First ISSN only
                    issn = source.get('issn')
                    if isinstance(issn, list) and len(issn) > 0:
                        result[f'location_{idx}_issn'] = issn[0]
                    else:
                        result[f'location_{idx}_issn'] = ''
                else:
                    result[f'location_{idx}_source'] = ''
                    result[f'location_{idx}_issn'] = ''
                
                # OA status
                result[f'location_{idx}_is_oa'] = location.get('is_oa') or False
            else:
                # Empty location
                result[f'location_{idx}_source'] = ''
                result[f'location_{idx}_issn'] = ''
                result[f'location_{idx}_is_oa'] = False
        
        return result
    
    def extract_sdgs(self, sdgs: List[Dict]) -> Dict:
        """Extract SDG information (top SDG)"""
        if sdgs and len(sdgs) > 0:
            sdg = sdgs[0]
            # Extract number from URL like https://metadata.un.org/sdg/16
            sdg_id = sdg.get('id', '')
            sdg_number = sdg_id.split('/')[-1] if sdg_id else ''
            
            return {
                'sdg_id': sdg_id,
                'sdg_number': sdg_number,
                'sdg_name': sdg.get('display_name', ''),
                'sdg_score': sdg.get('score', 0)
            }
        return {
            'sdg_id': '',
            'sdg_number': '',
            'sdg_name': '',
            'sdg_score': 0
        }
    
    def extract_counts_by_year(self, counts: List[Dict]) -> Dict:
        """Extract citation counts by year"""
        result = {'counts_by_year_json': json.dumps(counts) if counts else ''}
        
        # Also extract recent years individually for easier analysis
        if counts:
            # Create dict for easy lookup
            counts_dict = {c['year']: c['cited_by_count'] for c in counts}
            
            # Get last 5 years
            for year in range(2025, 2018, -1):
                result[f'citations_{year}'] = counts_dict.get(year, 0)
        else:
            for year in range(2025, 2018, -1):
                result[f'citations_{year}'] = 0
        
        return result
    
    def extract_work_data(self, data: Dict) -> Dict:
        """Extract all relevant data from a work"""
        if not data:
            return {}
        
        # Basic information
        record = {
            'id': data.get('id', ''),
            'doi': data.get('doi', ''),
            'title': data.get('title', ''),
            'publication_year': data.get('publication_year', ''),
            'publication_date': data.get('publication_date', ''),
            'language': data.get('language', ''),
            'type': data.get('type', ''),
            'is_retracted': data.get('is_retracted', False),
            'is_paratext': data.get('is_paratext', False),
        }
        
        # IDs
        ids = data.get('ids', {})
        record['openalex_id'] = ids.get('openalex', '')
        record['doi_id'] = ids.get('doi', '')
        record['pmid'] = ids.get('pmid', '')
        record['pmcid'] = ids.get('pmcid', '')
        record['mag'] = ids.get('mag', '')
        
        # Primary location - detailed source information
        primary_location = data.get('primary_location', {})
        if primary_location:
            record['is_oa'] = primary_location.get('is_oa', False)
            record['landing_page_url'] = primary_location.get('landing_page_url', '')
            record['pdf_url'] = primary_location.get('pdf_url', '')
            record['is_accepted'] = primary_location.get('is_accepted', False)
            record['is_published'] = primary_location.get('is_published', False)
            record['version'] = primary_location.get('version', '')
            record['license'] = primary_location.get('license', '')
            
            source = primary_location.get('source')
            if source and isinstance(source, dict):
                record['source_id'] = source.get('id') or ''
                record['source_name'] = source.get('display_name') or ''
                record['source_issn_l'] = source.get('issn_l') or ''
                
                # Get first ISSN only
                issn = source.get('issn')
                if isinstance(issn, list) and len(issn) > 0:
                    record['source_issn'] = issn[0]
                else:
                    record['source_issn'] = ''
                
                # Additional source metadata
                record['source_type'] = source.get('type', '')
                record['source_is_oa'] = source.get('is_oa', False)
                record['source_is_in_doaj'] = source.get('is_in_doaj', False)
                record['source_is_core'] = source.get('is_core', False)
                record['host_organization'] = source.get('host_organization', '')
                record['host_organization_name'] = source.get('host_organization_name', '')
            else:
                # No source available
                record['source_id'] = ''
                record['source_name'] = ''
                record['source_issn_l'] = ''
                record['source_issn'] = ''
                record['source_type'] = ''
                record['source_is_oa'] = False
                record['source_is_in_doaj'] = False
                record['source_is_core'] = False
                record['host_organization'] = ''
                record['host_organization_name'] = ''
        else:
            # No primary location
            for key in ['is_oa', 'is_accepted', 'is_published', 'source_is_oa', 'source_is_in_doaj', 'source_is_core']:
                record[key] = False
            for key in ['landing_page_url', 'pdf_url', 'version', 'license', 'source_id', 'source_name', 
                       'source_issn_l', 'source_issn', 'source_type', 'host_organization', 'host_organization_name']:
                record[key] = ''
        
        # Open access information
        open_access = data.get('open_access', {})
        record['oa_status'] = open_access.get('oa_status', '')
        record['oa_url'] = open_access.get('oa_url', '')
        record['any_repository_has_fulltext'] = open_access.get('any_repository_has_fulltext', False)
        
        # Indexed in
        record['indexed_in'] = '; '.join(data.get('indexed_in', []))
        
        # Biblio
        biblio = data.get('biblio', {})
        record['volume'] = biblio.get('volume', '')
        record['issue'] = biblio.get('issue', '')
        record['first_page'] = biblio.get('first_page', '')
        record['last_page'] = biblio.get('last_page', '')
        
        # Authors and affiliations
        authors_data = self.extract_authors_data(data.get('authorships', []))
        record.update(authors_data)
        
        # Topics
        topics_data = self.extract_topics_data(data)
        record.update(topics_data)
        
        # Keywords
        keywords_data = self.extract_keywords(data.get('keywords', []))
        record.update(keywords_data)
        
        # Concepts
        record['concepts'] = self.extract_concepts(data.get('concepts', []))
        
        # Locations
        locations_data = self.extract_locations(data.get('locations', []))
        record.update(locations_data)
        
        # Citations
        record['cited_by_count'] = data.get('cited_by_count', 0)
        
        # Citation percentiles
        citation_percentile = data.get('citation_normalized_percentile', {})
        record['citation_percentile_value'] = citation_percentile.get('value', 0) if citation_percentile else 0
        record['citation_is_top_1_percent'] = citation_percentile.get('is_in_top_1_percent', False) if citation_percentile else False
        record['citation_is_top_10_percent'] = citation_percentile.get('is_in_top_10_percent', False) if citation_percentile else False
        
        cited_by_percentile = data.get('cited_by_percentile_year', {})
        record['cited_by_percentile_min'] = cited_by_percentile.get('min', 0) if cited_by_percentile else 0
        record['cited_by_percentile_max'] = cited_by_percentile.get('max', 0) if cited_by_percentile else 0
        
        # FWCI (Field-Weighted Citation Impact)
        record['fwci'] = data.get('fwci', 0)
        
        # Counts by year
        counts_data = self.extract_counts_by_year(data.get('counts_by_year', []))
        record.update(counts_data)
        
        # Referenced works
        referenced_works = data.get('referenced_works', [])
        record['referenced_works_count'] = len(referenced_works)
        record['referenced_works'] = '; '.join(referenced_works)
        
        # Related works
        related_works = data.get('related_works', [])
        record['related_works_count'] = len(related_works)
        record['related_works'] = '; '.join(related_works)
        
        # SDGs
        sdgs_data = self.extract_sdgs(data.get('sustainable_development_goals', []))
        record.update(sdgs_data)
        
        # Grants
        grants = data.get('grants', [])
        record['grants_count'] = len(grants)
        if grants:
            grant_info = []
            for grant in grants[:5]:  # First 5 grants
                funder = grant.get('funder_display_name', '')
                award_id = grant.get('award_id', '')
                if funder:
                    grant_info.append(f"{funder} ({award_id})" if award_id else funder)
            record['grants'] = '; '.join(grant_info)
        else:
            record['grants'] = ''
        
        # APC
        apc = data.get('apc_paid', {})
        record['apc_value'] = apc.get('value', 0) if apc else 0
        record['apc_currency'] = apc.get('currency', '') if apc else ''
        
        return record
    
    def process_all_works(self):
        """Process all works from the CSV file"""
        print("="*70)
        print("OPENALEX API DATA EXTRACTOR - TEST MODE")
        print("="*70)
        
        # Read input CSV
        print(f"\n📂 Reading input file...")
        print(f"   {self.input_csv_path}")
        
        try:
            df_input = pd.read_csv(self.input_csv_path)
            
            # Test mode: only first N papers
            if self.test_mode:
                df_input = df_input.head(self.test_size)
                print(f"\n⚠️  TEST MODE: Processing only first {self.test_size} papers")
        except Exception as e:
            print(f"\n❌ Error reading CSV: {e}")
            return
        
        print(f"\n✓ Found {len(df_input)} works to process")
        print(f"{'='*70}\n")

        # Process each work
        failed_count = 0
        checkpoint_interval = 100  # Save every 100 papers

        for idx, row in df_input.iterrows():
            work_id = row['Node_ID']
            print(f"\n{'─'*70}")
            print(f"[{idx+1}/{len(df_input)}] Processing: {work_id}")
            
            # Fetch data
            data = self.fetch_work_data(work_id)
            
            if data:
                try:
                    # Extract data
                    record = self.extract_work_data(data)
                    if record:
                        self.all_results.append(record)
                        title = record.get('title') or 'No title'
                        # Safely handle None or empty titles
                        title_display = title[:80] if title else 'No title'
                        print(f"  📄 Title: {title_display}...")
                        print(f"  👥 Authors: {record.get('authors_count', 0)}")
                        print(f"  📊 Citations: {record.get('cited_by_count', 0)}")
                        print(f"  🎯 Primary topic: {record.get('primary_topic') or 'N/A'}")
                        print(f"  ✅ Successfully extracted all fields")
                    else:
                        failed_count += 1
                        print(f"  ⚠ Failed to extract data")
                except Exception as e:
                    failed_count += 1
                    print(f"  ❌ Error extracting data: {e}")
            else:
                failed_count += 1

            # Checkpoint save every N papers
            if len(self.all_results) > 0 and len(self.all_results) % checkpoint_interval == 0:
                print(f"\n  💾 Checkpoint: Saving {len(self.all_results)} records so far...")
                try:
                    checkpoint_df = pd.DataFrame(self.all_results)
                    checkpoint_path = self.input_csv_path.replace('.csv', f'_checkpoint_{len(self.all_results)}.xlsx')
                    checkpoint_df.to_excel(checkpoint_path, index=False)
                    print(f"  ✓ Checkpoint saved to: {checkpoint_path}")
                except Exception as e:
                    print(f"  ⚠ Could not save checkpoint: {e}")

            # Be respectful to the API - 1 second delay
            if idx < len(df_input) - 1:  # Don't delay after last item
                print(f"  ⏳ Waiting 1 second...")
                time.sleep(1)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successfully retrieved: {len(self.all_results)}")
        print(f"❌ Failed retrievals: {failed_count}")
    
    def create_dataset(self) -> pd.DataFrame:
        """Create comprehensive DataFrame from results"""
        if not self.all_results:
            print("⚠ No results to process.")
            return pd.DataFrame()
        
        print("\n📊 Creating dataset...")
        df = pd.DataFrame(self.all_results)
        print(f"✓ Created dataset with {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def save_to_excel(self, df: pd.DataFrame, output_path: str):
        """Save DataFrame to Excel with proper formatting"""
        print(f"\n💾 Saving to Excel...")
        print(f"   {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='OpenAlex Data', index=False)
            
            # Access worksheet
            worksheet = writer.sheets['OpenAlex Data']
            
            # Auto-adjust column widths
            for idx, col in enumerate(df.columns):
                # Calculate maximum length needed
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                
                # Special handling for long text columns
                if col.lower() in ['title', 'referenced_works', 'related_works', 'abstract']:
                    column_width = min(max_length + 2, 100)
                elif col.lower() in ['authors_names', 'institutions', 'concepts']:
                    column_width = min(max_length + 2, 80)
                else:
                    column_width = min(max_length + 2, 50)
                
                # Use proper Excel column letter
                column_letter = get_column_letter(idx + 1)
                worksheet.column_dimensions[column_letter].width = column_width
        
        print(f"✓ Successfully saved {len(df)} records")


def main():
    """
    Main execution function - TEST VERSION
    """
    # Define paths
    INPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\00_OA_corpus_LCC_diagnostic.csv"
    OUTPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\03_OpenAlex_API_CS_FULL.xlsx"
    
    print("\n" + "="*70)
    print("OPENALEX METADATA EXTRACTION - TEST RUN")
    print("Testing with first 10 papers")
    print("="*70)
    
    # Initialize extractor in TEST MODE
    # extractor = OpenAlexDataExtractor(INPUT_PATH, test_mode=True, test_size=10)
    extractor = OpenAlexDataExtractor(INPUT_PATH, test_mode=False)  # Change to False to process all papers
    
    # Process works
    extractor.process_all_works()
    
    if not extractor.all_results:
        print("\n❌ No data retrieved. Please check:")
        print("   - Input file path and format")
        print("   - Column name is 'Node_ID'")
        print("   - Internet connection")
        print("   - OpenAlex API access")
        return
    
    # Create dataset
    df = extractor.create_dataset()
    
    # Save to Excel
    extractor.save_to_excel(df, OUTPUT_PATH)
    
    # Display comprehensive summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📚 Basic Info:")
    print(f"   Records processed: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    
    print(f"\n📋 Sample of extracted data:")
    print(f"   Titles extracted: {df['title'].notna().sum()}")
    print(f"   DOIs found: {df['doi'].notna().sum()}")
    print(f"   Authors extracted: {df['authors_count'].sum()}")
    print(f"   Open Access papers: {df['is_oa'].sum()}")
    
    print(f"\n📊 Citation stats:")
    print(f"   Total citations: {df['cited_by_count'].sum()}")
    print(f"   Average: {df['cited_by_count'].mean():.1f}")
    
    print(f"\n🎯 Topic coverage:")
    print(f"   Papers with primary topic: {df['primary_topic'].notna().sum()}")
    print(f"   Papers with keywords: {df['keywords_all'].notna().sum()}")
    
    print(f"\n🌍 SDG info:")
    print(f"   Papers with SDG: {(df['sdg_number'] != '').sum()}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    print(f"\nTest dataset saved at:")
    print(f"{OUTPUT_PATH}")
    print(f"\n💡 If the test looks good, run the full version to process all 5252 papers!")
    
    # Show first few rows of key columns for verification
    print(f"\n📋 Preview of extracted data:")
    key_cols = ['title', 'publication_year', 'authors_count', 'cited_by_count', 'primary_topic', 'is_oa']
    available_cols = [col for col in key_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].head().to_string())


if __name__ == "__main__":
    main()