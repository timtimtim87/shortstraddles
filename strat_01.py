import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

class TakeProfitStraddleAnalyzer:
    def __init__(self, straddle_data_dir: str, output_dir: str = None, take_profit_pct: float = 50.0):
        """
        Initialize the take profit straddle analyzer.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            output_dir: Directory to save results
            take_profit_pct: Take profit percentage (default 50%)
        """
        self.straddle_data_dir = straddle_data_dir
        self.output_dir = output_dir or f"{straddle_data_dir}_tp_analysis"
        self.take_profit_pct = take_profit_pct
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.results = []
    
    def analyze_straddle_with_tp(self, filepath: str) -> dict:
        """Analyze straddle with take profit strategy."""
        filename = os.path.basename(filepath)
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get valid price data
            valid_data = df.dropna(subset=['straddle_price']).copy()
            
            if len(valid_data) < 2:
                return {
                    'filename': filename,
                    'error': 'Insufficient price data'
                }
            
            # Entry details
            entry_price = valid_data['straddle_price'].iloc[0]
            entry_date = valid_data['date'].iloc[0]
            
            if entry_price <= 0:
                return {
                    'filename': filename,
                    'error': 'Invalid entry price'
                }
            
            # Calculate take profit target
            # For short position: TP when price drops by take_profit_pct
            # TP Price = Entry Price * (1 - take_profit_pct/100)
            tp_price = entry_price * (1 - self.take_profit_pct / 100)
            
            # Parse contract info
            parts = filename.replace('_STRADDLE.csv', '').split('_')
            symbol = parts[0] if len(parts) > 0 else 'UNKNOWN'
            expiry = parts[1] if len(parts) > 1 else 'UNKNOWN'
            strike = float(parts[2]) if len(parts) > 2 else 0
            
            # Check if TP was hit
            tp_hit_data = valid_data[valid_data['straddle_price'] <= tp_price]
            
            if not tp_hit_data.empty:
                # TP was hit - find first occurrence
                tp_hit_row = tp_hit_data.iloc[0]
                exit_date = tp_hit_row['date']
                exit_price = tp_hit_row['straddle_price']
                exit_reason = 'take_profit'
                days_held = (exit_date - entry_date).days
                
                # Calculate actual return
                profit = entry_price - exit_price
                actual_return_pct = (profit / entry_price) * 100
                
            else:
                # TP was never hit - hold to end
                exit_date = valid_data['date'].iloc[-1]
                exit_price = valid_data['straddle_price'].iloc[-1]
                exit_reason = 'held_to_end'
                days_held = (exit_date - entry_date).days
                
                # Calculate actual return
                profit = entry_price - exit_price
                actual_return_pct = (profit / entry_price) * 100
            
            return {
                'filename': filename,
                'symbol': symbol,
                'expiry_date': expiry,
                'strike': strike,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'entry_price': entry_price,
                'tp_target_price': tp_price,
                'tp_target_pct': self.take_profit_pct,
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'tp_hit': exit_reason == 'take_profit',
                'days_held': days_held,
                'profit': profit,
                'actual_return_pct': actual_return_pct,
                'total_data_points': len(valid_data)
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e)
            }
    
    def process_all_files(self):
        """Process all straddle files."""
        self.logger.info(f"Processing all straddle files with {self.take_profit_pct}% take profit...")
        
        # Find all straddle files
        pattern = os.path.join(self.straddle_data_dir, "*_STRADDLE.csv")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.error(f"No straddle files found in {self.straddle_data_dir}")
            return
        
        self.logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        for filepath in files:
            result = self.analyze_straddle_with_tp(filepath)
            self.results.append(result)
            
            if 'error' not in result:
                status = "TP HIT" if result['tp_hit'] else "HELD"
                self.logger.info(f"Processed: {result['filename']} - {status} - Return: {result['actual_return_pct']:.2f}%")
            else:
                self.logger.error(f"Error processing {result['filename']}: {result['error']}")
        
        self.logger.info(f"Processing complete. {len(self.results)} files processed")
    
    def save_results(self):
        """Save results to CSV."""
        if not self.results:
            self.logger.warning("No results to save")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save detailed results
        output_path = os.path.join(self.output_dir, f'tp_{int(self.take_profit_pct)}pct_results_{timestamp}.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            print("No results to summarize")
            return
        
        # Filter valid results
        valid_results = [r for r in self.results if 'error' not in r]
        error_results = [r for r in self.results if 'error' in r]
        
        if not valid_results:
            print("No valid results found")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(valid_results)
        
        # Calculate statistics
        total_contracts = len(valid_results)
        tp_hit_count = len(df[df['tp_hit'] == True])
        tp_hit_rate = (tp_hit_count / total_contracts) * 100
        held_to_end_count = total_contracts - tp_hit_count
        
        profitable_count = len(df[df['actual_return_pct'] > 0])
        profitable_rate = (profitable_count / total_contracts) * 100
        
        avg_return = df['actual_return_pct'].mean()
        median_return = df['actual_return_pct'].median()
        min_return = df['actual_return_pct'].min()
        max_return = df['actual_return_pct'].max()
        std_return = df['actual_return_pct'].std()
        
        # Statistics by exit reason
        tp_hit_df = df[df['tp_hit'] == True]
        held_df = df[df['tp_hit'] == False]
        
        if not tp_hit_df.empty:
            avg_days_tp = tp_hit_df['days_held'].mean()
            avg_return_tp = tp_hit_df['actual_return_pct'].mean()
        else:
            avg_days_tp = 0
            avg_return_tp = 0
        
        if not held_df.empty:
            avg_days_held = held_df['days_held'].mean()
            avg_return_held = held_df['actual_return_pct'].mean()
        else:
            avg_days_held = 0
            avg_return_held = 0
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SHORT STRADDLE WITH {self.take_profit_pct}% TAKE PROFIT ANALYSIS")
        print(f"{'='*70}")
        print(f"Total contracts processed: {total_contracts}")
        print(f"Errors encountered: {len(error_results)}")
        print()
        print("TAKE PROFIT STATISTICS:")
        print(f"Take profit hit: {tp_hit_count}/{total_contracts} ({tp_hit_rate:.1f}%)")
        print(f"Held to end: {held_to_end_count}/{total_contracts} ({100-tp_hit_rate:.1f}%)")
        print()
        print("RETURN STATISTICS:")
        print(f"Profitable contracts: {profitable_count}/{total_contracts} ({profitable_rate:.1f}%)")
        print(f"Average return: {avg_return:.2f}%")
        print(f"Median return: {median_return:.2f}%")
        print(f"Standard deviation: {std_return:.2f}%")
        print(f"Return range: {min_return:.2f}% to {max_return:.2f}%")
        print()
        print("BY EXIT REASON:")
        if tp_hit_count > 0:
            print(f"Take Profit Hit ({tp_hit_count} contracts):")
            print(f"  Average return: {avg_return_tp:.2f}%")
            print(f"  Average days to TP: {avg_days_tp:.1f} days")
        
        if held_to_end_count > 0:
            print(f"Held to End ({held_to_end_count} contracts):")
            print(f"  Average return: {avg_return_held:.2f}%")
            print(f"  Average days held: {avg_days_held:.1f} days")
        print()
        
        # Top and bottom performers
        top_5 = df.nlargest(5, 'actual_return_pct')[['symbol', 'actual_return_pct', 'tp_hit', 'days_held']]
        bottom_5 = df.nsmallest(5, 'actual_return_pct')[['symbol', 'actual_return_pct', 'tp_hit', 'days_held']]
        
        print("TOP 5 PERFORMERS:")
        for _, row in top_5.iterrows():
            tp_status = "TP" if row['tp_hit'] else "HELD"
            print(f"  {row['symbol']}: {row['actual_return_pct']:.2f}% ({tp_status}, {row['days_held']} days)")
        
        print("\nWORST 5 PERFORMERS:")
        for _, row in bottom_5.iterrows():
            tp_status = "TP" if row['tp_hit'] else "HELD"
            print(f"  {row['symbol']}: {row['actual_return_pct']:.2f}% ({tp_status}, {row['days_held']} days)")
        
        print()
        print("STRATEGY EFFECTIVENESS:")
        if tp_hit_rate > 50:
            print(f"✓ High TP hit rate ({tp_hit_rate:.1f}%) - {self.take_profit_pct}% target frequently achieved")
        else:
            print(f"⚠ Low TP hit rate ({tp_hit_rate:.1f}%) - {self.take_profit_pct}% target rarely achieved")
        
        if avg_return > 0:
            print(f"✓ Positive average return ({avg_return:.2f}%)")
        else:
            print(f"⚠ Negative average return ({avg_return:.2f}%)")
        
        print(f"\n{'='*70}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("="*70)
        print(f"SHORT STRADDLE WITH {self.take_profit_pct}% TAKE PROFIT ANALYZER")
        print("="*70)
        print(f"Strategy: Enter short straddle on first day, exit at {self.take_profit_pct}% profit or hold to end")
        print("No stop loss - unlimited downside risk")
        print(f"Input directory: {self.straddle_data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Process files
        self.process_all_files()
        
        # Save results
        output_path = self.save_results()
        
        # Print summary
        self.print_summary()
        
        if output_path:
            print(f"\nDetailed results saved to: {output_path}")

def main():
    # Configuration
    straddle_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/tp_50_analysis"
    take_profit_pct = 50.0  # 50% take profit
    
    # Create analyzer and run
    analyzer = TakeProfitStraddleAnalyzer(
        straddle_data_dir=straddle_data_dir,
        output_dir=output_dir,
        take_profit_pct=take_profit_pct
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()