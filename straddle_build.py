import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

class StraddleFileGenerator:
    def __init__(self, enhanced_data_dir: str, output_dir: str = None):
        """
        Initialize the straddle file generator.
        
        Args:
            enhanced_data_dir: Directory containing enhanced CSV files
            output_dir: Directory to save straddle files (defaults to enhanced_data_dir + '_straddles')
        """
        self.enhanced_data_dir = enhanced_data_dir
        self.output_dir = output_dir or f"{enhanced_data_dir}_straddles"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_contract_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare contract data."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def find_matching_contracts(self, enhanced_data_dir: str) -> dict:
        """Find all matching call/put pairs."""
        # Get all enhanced CSV files
        pattern = os.path.join(enhanced_data_dir, "*_enhanced.csv")
        all_files = glob.glob(pattern)
        
        # Separate calls and puts
        call_files = [f for f in all_files if '_CALL_enhanced.csv' in f]
        put_files = [f for f in all_files if '_PUT_enhanced.csv' in f]
        
        # Create mapping
        matching_pairs = {}
        
        for call_file in call_files:
            # Extract base name (symbol_expiry_strike)
            basename = os.path.basename(call_file).replace('_CALL_enhanced.csv', '')
            
            # Find matching put
            put_file = os.path.join(enhanced_data_dir, f"{basename}_PUT_enhanced.csv")
            
            if put_file in put_files:
                matching_pairs[basename] = {
                    'call_file': call_file,
                    'put_file': put_file
                }
                
        self.logger.info(f"Found {len(matching_pairs)} matching call/put pairs")
        return matching_pairs
    
    def calculate_straddle_metrics(self, call_df: pd.DataFrame, put_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive straddle metrics from call and put data."""
        # Merge on date
        straddle_df = pd.merge(
            call_df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions',
                     'symbol', 'expiry_date', 'strike', 'spot_price', 'time_to_expiry', 'iv', 'opp', 'moneyness']], 
            put_df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'iv', 'opp']], 
            on='date', 
            suffixes=('_call', '_put')
        )
        
        # Calculate basic straddle metrics
        straddle_df['straddle_open'] = straddle_df['open_call'] + straddle_df['open_put']
        straddle_df['straddle_high'] = straddle_df['high_call'] + straddle_df['high_put']
        straddle_df['straddle_low'] = straddle_df['low_call'] + straddle_df['low_put']
        straddle_df['straddle_close'] = straddle_df['close_call'] + straddle_df['close_put']
        straddle_df['straddle_volume'] = straddle_df['volume_call'] + straddle_df['volume_put']
        straddle_df['straddle_transactions'] = straddle_df['transactions_call'] + straddle_df['transactions_put']
        
        # Calculate weighted average VWAP
        total_volume = straddle_df['volume_call'] + straddle_df['volume_put']
        mask = total_volume > 0
        straddle_df.loc[mask, 'straddle_vwap'] = (
            (straddle_df.loc[mask, 'vwap_call'] * straddle_df.loc[mask, 'volume_call'] + 
             straddle_df.loc[mask, 'vwap_put'] * straddle_df.loc[mask, 'volume_put']) / 
            total_volume.loc[mask]
        )
        straddle_df.loc[~mask, 'straddle_vwap'] = np.nan
        
        # Calculate price-weighted IV
        straddle_df['straddle_price'] = straddle_df['straddle_close']  # Alias for consistency
        
        # Price-weighted IV calculation
        mask = (straddle_df['straddle_price'] > 0) & (straddle_df['iv_call'].notna()) & (straddle_df['iv_put'].notna())
        straddle_df.loc[mask, 'straddle_iv'] = (
            (straddle_df.loc[mask, 'close_call'] * straddle_df.loc[mask, 'iv_call'] + 
             straddle_df.loc[mask, 'close_put'] * straddle_df.loc[mask, 'iv_put']) / 
            straddle_df.loc[mask, 'straddle_price']
        )
        
        # For cases where one IV is missing, use the available one
        mask_call_only = (straddle_df['iv_call'].notna()) & (straddle_df['iv_put'].isna())
        straddle_df.loc[mask_call_only, 'straddle_iv'] = straddle_df.loc[mask_call_only, 'iv_call']
        
        mask_put_only = (straddle_df['iv_call'].isna()) & (straddle_df['iv_put'].notna())
        straddle_df.loc[mask_put_only, 'straddle_iv'] = straddle_df.loc[mask_put_only, 'iv_put']
        
        # Calculate straddle OPP (Options Price Percentage)
        mask = (straddle_df['spot_price'] > 0) & (straddle_df['straddle_price'] > 0)
        straddle_df.loc[mask, 'straddle_opp'] = (straddle_df.loc[mask, 'straddle_price'] / straddle_df.loc[mask, 'spot_price']) * 100
        
        # Calculate Greeks ratios and differences
        straddle_df['iv_spread'] = abs(straddle_df['iv_call'] - straddle_df['iv_put'])
        straddle_df['price_ratio'] = straddle_df['close_call'] / (straddle_df['close_put'] + 1e-10)  # Avoid division by zero
        straddle_df['volume_ratio'] = straddle_df['volume_call'] / (straddle_df['volume_put'] + 1e-10)
        
        # Calculate daily returns
        straddle_df['straddle_return'] = straddle_df['straddle_close'].pct_change()
        straddle_df['call_return'] = straddle_df['close_call'].pct_change()
        straddle_df['put_return'] = straddle_df['close_put'].pct_change()
        
        # Clean up column order
        columns_order = [
            'date', 'symbol', 'expiry_date', 'strike', 'spot_price', 'time_to_expiry', 'moneyness',
            'straddle_open', 'straddle_high', 'straddle_low', 'straddle_close', 'straddle_price',
            'straddle_volume', 'straddle_vwap', 'straddle_transactions', 'straddle_iv', 'straddle_opp',
            'straddle_return', 'iv_spread', 'price_ratio', 'volume_ratio',
            'ticker_call', 'open_call', 'high_call', 'low_call', 'close_call', 'volume_call', 
            'vwap_call', 'transactions_call', 'iv_call', 'opp_call', 'call_return',
            'ticker_put', 'open_put', 'high_put', 'low_put', 'close_put', 'volume_put', 
            'vwap_put', 'transactions_put', 'iv_put', 'opp_put', 'put_return'
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in columns_order if col in straddle_df.columns]
        remaining_columns = [col for col in straddle_df.columns if col not in existing_columns]
        
        final_columns = existing_columns + remaining_columns
        straddle_df = straddle_df[final_columns]
        
        return straddle_df
    
    def create_straddle_file(self, contract_pair: dict, base_name: str):
        """Create a straddle file from a call/put pair."""
        self.logger.info(f"Processing straddle: {base_name}")
        
        # Load call and put data
        call_df = self.load_contract_data(contract_pair['call_file'])
        put_df = self.load_contract_data(contract_pair['put_file'])
        
        if call_df is None or put_df is None:
            self.logger.error(f"Could not load data for {base_name}")
            return False
        
        # Calculate straddle metrics
        straddle_df = self.calculate_straddle_metrics(call_df, put_df)
        
        if straddle_df.empty:
            self.logger.warning(f"No overlapping data for {base_name}")
            return False
        
        # Save straddle file
        filename = f"{base_name}_STRADDLE.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            straddle_df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {filename} with {len(straddle_df)} rows")
            
            # Log some summary statistics
            avg_straddle_price = straddle_df['straddle_price'].mean()
            avg_straddle_iv = straddle_df['straddle_iv'].mean()
            avg_straddle_opp = straddle_df['straddle_opp'].mean()
            
            self.logger.info(f"  Avg Price: ${avg_straddle_price:.2f}, Avg IV: {avg_straddle_iv:.3f}, Avg OPP: {avg_straddle_opp:.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {filename}: {e}")
            return False
    
    def generate_all_straddle_files(self):
        """Generate straddle files for all matching call/put pairs."""
        self.logger.info("Starting straddle file generation")
        
        # Find all matching pairs
        matching_pairs = self.find_matching_contracts(self.enhanced_data_dir)
        
        if not matching_pairs:
            self.logger.error("No matching call/put pairs found")
            return
        
        successful_files = 0
        total_files = len(matching_pairs)
        
        for base_name, contract_pair in matching_pairs.items():
            try:
                success = self.create_straddle_file(contract_pair, base_name)
                if success:
                    successful_files += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {base_name}: {e}")
        
        self.logger.info(f"Successfully created {successful_files}/{total_files} straddle files")
        
        # Create summary report
        self.create_straddle_summary()
    
    def create_straddle_summary(self):
        """Create a summary report of all straddle files."""
        try:
            # Find all straddle files
            pattern = os.path.join(self.output_dir, "*_STRADDLE.csv")
            straddle_files = glob.glob(pattern)
            
            if not straddle_files:
                self.logger.warning("No straddle files found for summary")
                return
            
            summary_data = []
            
            for filepath in straddle_files:
                try:
                    df = pd.read_csv(filepath)
                    
                    if len(df) == 0:
                        continue
                    
                    # Extract info
                    filename = os.path.basename(filepath)
                    parts = filename.replace('_STRADDLE.csv', '').split('_')
                    
                    stats = {
                        'file': filename,
                        'symbol': parts[0],
                        'expiry_date': parts[1],
                        'strike': float(parts[2]),
                        'total_rows': len(df),
                        'avg_straddle_price': df['straddle_price'].mean(),
                        'median_straddle_price': df['straddle_price'].median(),
                        'max_straddle_price': df['straddle_price'].max(),
                        'min_straddle_price': df['straddle_price'].min(),
                        'avg_straddle_iv': df['straddle_iv'].mean(),
                        'avg_straddle_opp': df['straddle_opp'].mean(),
                        'avg_volume': df['straddle_volume'].mean(),
                        'total_volume': df['straddle_volume'].sum(),
                        'date_range': f"{df['date'].min()} to {df['date'].max()}"
                    }
                    
                    summary_data.append(stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {filepath} for summary: {e}")
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(self.output_dir, 'straddle_summary_report.csv')
                summary_df.to_csv(summary_path, index=False)
                
                self.logger.info(f"Straddle summary report saved to: {summary_path}")
                
                # Print summary to console
                print("\n" + "="*100)
                print("STRADDLE FILES SUMMARY")
                print("="*100)
                print(f"Total straddle files created: {len(summary_data)}")
                print()
                
                # Group by symbol
                for symbol in sorted(summary_df['symbol'].unique()):
                    symbol_data = summary_df[summary_df['symbol'] == symbol]
                    print(f"{symbol}: {len(symbol_data)} straddles")
                    
                    for _, row in symbol_data.iterrows():
                        print(f"  {row['expiry_date']} ${row['strike']:.0f}: "
                              f"Avg ${row['avg_straddle_price']:.2f}, "
                              f"IV {row['avg_straddle_iv']:.3f}, "
                              f"OPP {row['avg_straddle_opp']:.2f}%, "
                              f"Vol {row['total_volume']:,.0f}")
                    print()
            
        except Exception as e:
            self.logger.error(f"Error creating straddle summary: {e}")

def main():
    # Configuration
    enhanced_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/options_data_enhanced"
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    
    # Create generator and process files
    generator = StraddleFileGenerator(
        enhanced_data_dir=enhanced_data_dir,
        output_dir=output_dir
    )
    
    generator.generate_all_straddle_files()
    
    print(f"\nStraddle files saved to: {output_dir}")

if __name__ == "__main__":
    main()