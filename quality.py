import pandas as pd
import numpy as np
import os
import glob
import shutil
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Set

class StraddleDataCleaner:
    def __init__(self, enhanced_data_dir: str, straddle_data_dir: str = None, 
                 backup_dir: str = None, output_dir: str = None):
        """
        Initialize the straddle-focused data cleaner.
        
        Args:
            enhanced_data_dir: Directory containing enhanced CSV files
            straddle_data_dir: Directory containing straddle CSV files
            backup_dir: Directory to backup deleted files (if None, files are permanently deleted)
            output_dir: Directory to save cleanup reports
        """
        self.enhanced_data_dir = enhanced_data_dir
        self.straddle_data_dir = straddle_data_dir or f"{enhanced_data_dir}_straddles"
        self.backup_dir = backup_dir
        self.output_dir = output_dir or f"{enhanced_data_dir}_cleanup"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.backup_dir:
            os.makedirs(self.backup_dir, exist_ok=True)
            os.makedirs(os.path.join(self.backup_dir, 'enhanced'), exist_ok=True)
            os.makedirs(os.path.join(self.backup_dir, 'straddles'), exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.output_dir, f'straddle_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Simple criteria for straddle analysis
        self.cleanup_criteria = {
            'min_dte_at_start': 330,    # Must have data at least 350 DTE
            'min_data_points': 30       # Must have at least 30 data points
        }
        
        self.removal_stats = {
            'unpaired_contracts': 0,
            'insufficient_dte': 0,
            'insufficient_data_points': 0,
            'straddles_removed': 0,
            'total_removed': 0
        }
    
    def find_contract_pairs(self) -> Dict[str, Dict]:
        """Find all call/put pairs and identify unpaired contracts."""
        pattern = os.path.join(self.enhanced_data_dir, "*_enhanced.csv")
        all_files = glob.glob(pattern)
        
        pairs = {}
        unpaired = []
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            
            if '_CALL_enhanced.csv' in filename:
                base_name = filename.replace('_CALL_enhanced.csv', '')
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]['call'] = filepath
                
            elif '_PUT_enhanced.csv' in filename:
                base_name = filename.replace('_PUT_enhanced.csv', '')
                if base_name not in pairs:
                    pairs[base_name] = {}
                pairs[base_name]['put'] = filepath
        
        # Find complete pairs and unpaired contracts
        complete_pairs = {}
        for base_name, files in pairs.items():
            if 'call' in files and 'put' in files:
                complete_pairs[base_name] = files
            else:
                # This is an unpaired contract
                if 'call' in files:
                    unpaired.append(files['call'])
                if 'put' in files:
                    unpaired.append(files['put'])
        
        self.logger.info(f"Found {len(complete_pairs)} complete call/put pairs")
        self.logger.info(f"Found {len(unpaired)} unpaired contracts")
        
        return complete_pairs, unpaired
    
    def analyze_contract_quality(self, call_filepath: str, put_filepath: str) -> Dict:
        """Analyze if a call/put pair meets quality criteria."""
        try:
            # Load both files
            call_df = pd.read_csv(call_filepath)
            put_df = pd.read_csv(put_filepath)
            
            if call_df.empty or put_df.empty:
                return {
                    'keep': False,
                    'reason': 'Empty data file',
                    'call_points': len(call_df),
                    'put_points': len(put_df),
                    'call_max_dte': 0,
                    'put_max_dte': 0
                }
            
            # Parse contract info from call file
            call_df['date'] = pd.to_datetime(call_df['date'])
            put_df['date'] = pd.to_datetime(put_df['date'])
            
            # Get expiry date from call file
            expiry_date = call_df['expiry_date'].iloc[0]
            expiry_dt = pd.to_datetime(expiry_date)
            
            # Calculate DTE for each file
            call_df['dte'] = (expiry_dt - call_df['date']).dt.days
            put_df['dte'] = (expiry_dt - put_df['date']).dt.days
            
            call_max_dte = call_df['dte'].max()
            put_max_dte = put_df['dte'].max()
            call_points = len(call_df)
            put_points = len(put_df)
            
            # Check criteria
            reasons = []
            
            # Criterion 1: Must have at least 350 DTE
            if call_max_dte < self.cleanup_criteria['min_dte_at_start']:
                reasons.append(f"Call insufficient DTE ({call_max_dte} < {self.cleanup_criteria['min_dte_at_start']})")
            
            if put_max_dte < self.cleanup_criteria['min_dte_at_start']:
                reasons.append(f"Put insufficient DTE ({put_max_dte} < {self.cleanup_criteria['min_dte_at_start']})")
            
            # Criterion 2: Must have at least 30 data points
            if call_points < self.cleanup_criteria['min_data_points']:
                reasons.append(f"Call insufficient data points ({call_points} < {self.cleanup_criteria['min_data_points']})")
            
            if put_points < self.cleanup_criteria['min_data_points']:
                reasons.append(f"Put insufficient data points ({put_points} < {self.cleanup_criteria['min_data_points']})")
            
            return {
                'keep': len(reasons) == 0,
                'reason': '; '.join(reasons) if reasons else 'Meets all criteria',
                'call_points': call_points,
                'put_points': put_points,
                'call_max_dte': call_max_dte,
                'put_max_dte': put_max_dte
            }
            
        except Exception as e:
            return {
                'keep': False,
                'reason': f'Analysis error: {str(e)}',
                'call_points': 0,
                'put_points': 0,
                'call_max_dte': 0,
                'put_max_dte': 0
            }
    
    def backup_file(self, filepath: str, file_type: str = 'enhanced') -> bool:
        """Backup a file before deletion."""
        if not self.backup_dir:
            return True
        
        try:
            filename = os.path.basename(filepath)
            backup_path = os.path.join(self.backup_dir, file_type, filename)
            shutil.copy2(filepath, backup_path)
            return True
        except Exception as e:
            self.logger.error(f"Error backing up {filepath}: {e}")
            return False
    
    def remove_files(self, files_to_remove: List[str], reason: str, dry_run: bool = True):
        """Remove a list of files."""
        for filepath in files_to_remove:
            filename = os.path.basename(filepath)
            
            if dry_run:
                self.logger.info(f"WOULD REMOVE: {filename} - {reason}")
            else:
                try:
                    # Backup if requested
                    if self.backup_dir:
                        file_type = 'straddles' if '_STRADDLE.csv' in filename else 'enhanced'
                        if not self.backup_file(filepath, file_type):
                            self.logger.warning(f"Failed to backup {filename}, skipping removal")
                            continue
                    
                    # Remove the file
                    os.remove(filepath)
                    self.removal_stats['total_removed'] += 1
                    self.logger.info(f"REMOVED: {filename} - {reason}")
                    
                except Exception as e:
                    self.logger.error(f"Error removing {filepath}: {e}")
    
    def analyze_and_clean(self, dry_run: bool = True):
        """Analyze data and identify what needs to be cleaned."""
        self.logger.info(f"Starting straddle-focused cleanup analysis {'(DRY RUN)' if dry_run else '(LIVE RUN)'}")
        self.logger.info(f"Criteria: Min {self.cleanup_criteria['min_dte_at_start']} DTE, Min {self.cleanup_criteria['min_data_points']} data points")
        
        # Step 1: Find pairs and unpaired contracts
        complete_pairs, unpaired_contracts = self.find_contract_pairs()
        
        # Step 2: Analyze each complete pair
        pairs_to_remove = []
        pairs_analysis = []
        
        for base_name, files in complete_pairs.items():
            analysis = self.analyze_contract_quality(files['call'], files['put'])
            
            pair_info = {
                'base_name': base_name,
                'call_file': os.path.basename(files['call']),
                'put_file': os.path.basename(files['put']),
                'keep': analysis['keep'],
                'reason': analysis['reason'],
                'call_points': analysis['call_points'],
                'put_points': analysis['put_points'],
                'call_max_dte': analysis['call_max_dte'],
                'put_max_dte': analysis['put_max_dte']
            }
            
            pairs_analysis.append(pair_info)
            
            if not analysis['keep']:
                pairs_to_remove.append({
                    'base_name': base_name,
                    'call_file': files['call'],
                    'put_file': files['put'],
                    'reason': analysis['reason']
                })
                
                # Count removal reasons
                if 'insufficient DTE' in analysis['reason']:
                    self.removal_stats['insufficient_dte'] += 1
                if 'insufficient data points' in analysis['reason']:
                    self.removal_stats['insufficient_data_points'] += 1
        
        # Step 3: Prepare removal lists
        files_to_remove = []
        
        # Add unpaired contracts
        for filepath in unpaired_contracts:
            files_to_remove.append(filepath)
        self.removal_stats['unpaired_contracts'] = len(unpaired_contracts)
        
        # Add pairs that don't meet criteria
        for pair in pairs_to_remove:
            files_to_remove.extend([pair['call_file'], pair['put_file']])
            
            # Also remove corresponding straddle
            straddle_file = os.path.join(self.straddle_data_dir, f"{pair['base_name']}_STRADDLE.csv")
            if os.path.exists(straddle_file):
                files_to_remove.append(straddle_file)
                self.removal_stats['straddles_removed'] += 1
        
        # Step 4: Save analysis report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save pairs analysis
        if pairs_analysis:
            pairs_df = pd.DataFrame(pairs_analysis)
            pairs_report_path = os.path.join(self.output_dir, f'pairs_analysis_{timestamp}.csv')
            pairs_df.to_csv(pairs_report_path, index=False)
            self.logger.info(f"Pairs analysis saved to: {pairs_report_path}")
        
        # Save removal candidates
        removal_details = []
        
        # Unpaired contracts
        for filepath in unpaired_contracts:
            removal_details.append({
                'filename': os.path.basename(filepath),
                'type': 'Unpaired contract',
                'reason': 'No matching call/put pair'
            })
        
        # Pairs that don't meet criteria
        for pair in pairs_to_remove:
            removal_details.extend([
                {
                    'filename': os.path.basename(pair['call_file']),
                    'type': 'Call (paired removal)',
                    'reason': pair['reason']
                },
                {
                    'filename': os.path.basename(pair['put_file']),
                    'type': 'Put (paired removal)',
                    'reason': pair['reason']
                }
            ])
            
            # Straddle
            straddle_file = os.path.join(self.straddle_data_dir, f"{pair['base_name']}_STRADDLE.csv")
            if os.path.exists(straddle_file):
                removal_details.append({
                    'filename': f"{pair['base_name']}_STRADDLE.csv",
                    'type': 'Straddle (paired removal)',
                    'reason': pair['reason']
                })
        
        if removal_details:
            removal_df = pd.DataFrame(removal_details)
            removal_report_path = os.path.join(self.output_dir, f'removal_candidates_{timestamp}.csv')
            removal_df.to_csv(removal_report_path, index=False)
            self.logger.info(f"Removal candidates saved to: {removal_report_path}")
        
        # Step 5: Print summary
        self.print_analysis_summary(complete_pairs, unpaired_contracts, pairs_to_remove)
        
        # Step 6: Perform removal if requested
        if not dry_run:
            # Remove unpaired contracts
            if unpaired_contracts:
                self.remove_files(unpaired_contracts, "Unpaired contract", dry_run=False)
            
            # Remove pairs that don't meet criteria
            for pair in pairs_to_remove:
                files_to_remove_for_pair = [pair['call_file'], pair['put_file']]
                straddle_file = os.path.join(self.straddle_data_dir, f"{pair['base_name']}_STRADDLE.csv")
                if os.path.exists(straddle_file):
                    files_to_remove_for_pair.append(straddle_file)
                
                self.remove_files(files_to_remove_for_pair, pair['reason'], dry_run=False)
        
        return len(files_to_remove), pairs_analysis, removal_details
    
    def print_analysis_summary(self, complete_pairs, unpaired_contracts, pairs_to_remove):
        """Print analysis summary."""
        total_pairs = len(complete_pairs)
        pairs_to_keep = total_pairs - len(pairs_to_remove)
        
        print(f"\n{'='*80}")
        print("STRADDLE DATA CLEANUP ANALYSIS")
        print(f"{'='*80}")
        print(f"Complete call/put pairs found: {total_pairs}")
        print(f"Pairs meeting criteria: {pairs_to_keep}")
        print(f"Pairs to remove: {len(pairs_to_remove)}")
        print(f"Unpaired contracts to remove: {len(unpaired_contracts)}")
        print()
        
        print("REMOVAL BREAKDOWN:")
        print(f"  Unpaired contracts: {len(unpaired_contracts)}")
        print(f"  Pairs with insufficient DTE: {self.removal_stats['insufficient_dte']}")
        print(f"  Pairs with insufficient data points: {self.removal_stats['insufficient_data_points']}")
        print(f"  Straddle files to remove: {self.removal_stats['straddles_removed']}")
        print()
        
        if pairs_to_remove:
            print("SAMPLE PAIRS TO REMOVE:")
            for i, pair in enumerate(pairs_to_remove[:5]):  # Show first 5
                print(f"  {pair['base_name']}: {pair['reason']}")
            if len(pairs_to_remove) > 5:
                print(f"  ... and {len(pairs_to_remove) - 5} more")
        
        print(f"\nFINAL RESULT: {pairs_to_keep} complete, high-quality straddle pairs will remain")
        print(f"{'='*80}")

def main():
    # Configuration
    enhanced_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/options_data_enhanced"
    straddle_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    backup_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/backup_removed_files"  # Set to None to skip backup
    
    # Create cleaner
    cleaner = StraddleDataCleaner(
        enhanced_data_dir=enhanced_data_dir,
        straddle_data_dir=straddle_data_dir,
        backup_dir=backup_dir
    )
    
    print("="*80)
    print("STRADDLE-FOCUSED DATA CLEANER")
    print("="*80)
    print("This will remove contracts that don't meet straddle analysis requirements:")
    print("1. Must have both call AND put contracts")
    print("2. Must have data starting at least 350 DTE")
    print("3. Must have at least 30 data points")
    print(f"Backup directory: {backup_dir or 'No backup (permanent deletion)'}")
    print()
    
    # Run analysis
    print("Analyzing data...")
    total_to_remove, pairs_analysis, removal_details = cleaner.analyze_and_clean(dry_run=True)
    
    # Ask for confirmation
    print(f"\nThis will remove {total_to_remove} files.")
    response = input("Proceed with cleanup? (type 'yes' to confirm): ")
    
    if response.lower() == 'yes':
        print("Proceeding with cleanup...")
        cleaner.analyze_and_clean(dry_run=False)
        print("Cleanup completed!")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()