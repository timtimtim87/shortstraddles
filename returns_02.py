import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

class SimpleShortReturnsCalculator:
    def __init__(self, straddle_data_dir: str, output_dir: str = None):
        """
        Initialize the simple short returns calculator.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            output_dir: Directory to save results
        """
        self.straddle_data_dir = straddle_data_dir
        self.output_dir = output_dir or f"{straddle_data_dir}_simple_short_returns"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.results = []
    
    def calculate_short_return(self, filepath: str) -> dict:
        """Calculate simple short return from first to last price."""
        filename = os.path.basename(filepath)
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get valid price data
            valid_prices = df.dropna(subset=['straddle_price'])
            
            if len(valid_prices) < 2:
                return {
                    'filename': filename,
                    'error': 'Insufficient price data'
                }
            
            # Get first and last prices
            first_price = valid_prices['straddle_price'].iloc[0]
            last_price = valid_prices['straddle_price'].iloc[-1]
            first_date = valid_prices['date'].iloc[0]
            last_date = valid_prices['date'].iloc[-1]
            
            # Calculate short return
            # Short position: Sell at first_price, buy back at last_price
            # Profit = first_price - last_price
            # Return = (first_price - last_price) / first_price * 100
            profit = first_price - last_price
            return_pct = (profit / first_price) * 100
            
            # Parse contract info
            parts = filename.replace('_STRADDLE.csv', '').split('_')
            symbol = parts[0] if len(parts) > 0 else 'UNKNOWN'
            expiry = parts[1] if len(parts) > 1 else 'UNKNOWN'
            strike = float(parts[2]) if len(parts) > 2 else 0
            
            # Calculate days held
            days_held = (last_date - first_date).days
            
            return {
                'filename': filename,
                'symbol': symbol,
                'expiry_date': expiry,
                'strike': strike,
                'first_date': first_date.strftime('%Y-%m-%d'),
                'last_date': last_date.strftime('%Y-%m-%d'),
                'first_price': first_price,
                'last_price': last_price,
                'profit': profit,
                'return_pct': return_pct,
                'days_held': days_held,
                'total_data_points': len(valid_prices)
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e)
            }
    
    def process_all_files(self):
        """Process all straddle files."""
        self.logger.info("Processing all straddle files for simple short returns...")
        
        # Find all straddle files
        pattern = os.path.join(self.straddle_data_dir, "*_STRADDLE.csv")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.error(f"No straddle files found in {self.straddle_data_dir}")
            return
        
        self.logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        for filepath in files:
            result = self.calculate_short_return(filepath)
            self.results.append(result)
            
            if 'error' not in result:
                self.logger.info(f"Processed: {result['filename']} - Return: {result['return_pct']:.2f}%")
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
        output_path = os.path.join(self.output_dir, f'simple_short_returns_{timestamp}.csv')
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
        profitable_count = len(df[df['return_pct'] > 0])
        profitable_rate = (profitable_count / total_contracts) * 100
        
        avg_return = df['return_pct'].mean()
        median_return = df['return_pct'].median()
        min_return = df['return_pct'].min()
        max_return = df['return_pct'].max()
        std_return = df['return_pct'].std()
        
        avg_days = df['days_held'].mean()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SIMPLE SHORT STRADDLE RETURNS SUMMARY")
        print(f"{'='*60}")
        print(f"Total contracts processed: {total_contracts}")
        print(f"Errors encountered: {len(error_results)}")
        print()
        print("RETURN STATISTICS:")
        print(f"Profitable contracts: {profitable_count}/{total_contracts} ({profitable_rate:.1f}%)")
        print(f"Average return: {avg_return:.2f}%")
        print(f"Median return: {median_return:.2f}%")
        print(f"Standard deviation: {std_return:.2f}%")
        print(f"Return range: {min_return:.2f}% to {max_return:.2f}%")
        print()
        print("HOLDING PERIOD:")
        print(f"Average days held: {avg_days:.1f} days")
        print()
        
        # Top and bottom performers
        top_5 = df.nlargest(5, 'return_pct')[['filename', 'symbol', 'return_pct']]
        bottom_5 = df.nsmallest(5, 'return_pct')[['filename', 'symbol', 'return_pct']]
        
        print("TOP 5 PERFORMERS:")
        for _, row in top_5.iterrows():
            print(f"  {row['symbol']}: {row['return_pct']:.2f}%")
        
        print("\nWORST 5 PERFORMERS:")
        for _, row in bottom_5.iterrows():
            print(f"  {row['symbol']}: {row['return_pct']:.2f}%")
        
        # By symbol summary (if multiple contracts per symbol)
        if 'symbol' in df.columns:
            symbol_stats = df.groupby('symbol').agg({
                'return_pct': ['count', 'mean', 'std'],
                'days_held': 'mean'
            }).round(2)
            
            if len(symbol_stats) <= 20:  # Only show if reasonable number of symbols
                print(f"\nBY SYMBOL SUMMARY:")
                print("Symbol | Count | Avg Return | Std Dev | Avg Days")
                print("-" * 50)
                for symbol in symbol_stats.index:
                    count = symbol_stats.loc[symbol, ('return_pct', 'count')]
                    avg_ret = symbol_stats.loc[symbol, ('return_pct', 'mean')]
                    std_ret = symbol_stats.loc[symbol, ('return_pct', 'std')]
                    avg_days = symbol_stats.loc[symbol, ('days_held', 'mean')]
                    print(f"{symbol:6} | {count:5.0f} | {avg_ret:9.2f}% | {std_ret:7.2f}% | {avg_days:8.1f}")
        
        print(f"\n{'='*60}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("="*60)
        print("SIMPLE SHORT RETURNS CALCULATOR")
        print("="*60)
        print("Calculating first-price to last-price returns for short positions")
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
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/simple_short_returns"
    
    # Create calculator and run
    calculator = SimpleShortReturnsCalculator(
        straddle_data_dir=straddle_data_dir,
        output_dir=output_dir
    )
    
    calculator.run_analysis()

if __name__ == "__main__":
    main()