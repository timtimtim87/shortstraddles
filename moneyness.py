import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

class MoneynessTakeProfitAnalyzer:
    def __init__(self, straddle_data_dir: str, output_dir: str = None, take_profit_pct: float = 50.0):
        """
        Initialize the moneyness-based take profit analyzer.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            output_dir: Directory to save results
            take_profit_pct: Take profit percentage (default 50%)
        """
        self.straddle_data_dir = straddle_data_dir
        self.output_dir = output_dir or f"{straddle_data_dir}_moneyness_tp_analysis"
        self.take_profit_pct = take_profit_pct
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.results = []
    
    def categorize_moneyness(self, spot_price: float, strike_price: float) -> str:
        """Categorize straddle by moneyness."""
        if spot_price <= 0:
            return 'UNKNOWN'
        
        # Calculate percentage difference from spot to strike
        distance_pct = ((strike_price - spot_price) / spot_price) * 100
        
        # Categorize based on distance
        if abs(distance_pct) <= 2.5:
            return 'ATM'  # At-the-money: within 2.5%
        elif 2.5 < distance_pct <= 15:
            return 'OTM_CALL'  # Strike above spot (call side out-of-money)
        elif distance_pct > 15:
            return 'FAR_OTM_CALL'  # Strike well above spot
        elif -15 <= distance_pct < -2.5:
            return 'OTM_PUT'  # Strike below spot (put side out-of-money)
        else:  # distance_pct < -15
            return 'FAR_OTM_PUT'  # Strike well below spot
    
    def analyze_straddle_with_tp(self, filepath: str) -> dict:
        """Analyze straddle with take profit strategy and moneyness categorization."""
        filename = os.path.basename(filepath)
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get valid price data
            valid_data = df.dropna(subset=['straddle_price', 'spot_price']).copy()
            
            if len(valid_data) < 2:
                return {
                    'filename': filename,
                    'error': 'Insufficient price data'
                }
            
            # Entry details
            entry_row = valid_data.iloc[0]
            entry_price = entry_row['straddle_price']
            entry_date = entry_row['date']
            entry_spot_price = entry_row['spot_price']
            
            if entry_price <= 0 or entry_spot_price <= 0:
                return {
                    'filename': filename,
                    'error': 'Invalid entry prices'
                }
            
            # Parse contract info
            parts = filename.replace('_STRADDLE.csv', '').split('_')
            symbol = parts[0] if len(parts) > 0 else 'UNKNOWN'
            expiry = parts[1] if len(parts) > 1 else 'UNKNOWN'
            strike = float(parts[2]) if len(parts) > 2 else 0
            
            # Calculate moneyness
            moneyness_category = self.categorize_moneyness(entry_spot_price, strike)
            distance_pct = ((strike - entry_spot_price) / entry_spot_price) * 100
            
            # Calculate take profit target
            tp_price = entry_price * (1 - self.take_profit_pct / 100)
            
            # Check if TP was hit
            tp_hit_data = valid_data[valid_data['straddle_price'] <= tp_price]
            
            if not tp_hit_data.empty:
                # TP was hit - find first occurrence
                tp_hit_row = tp_hit_data.iloc[0]
                exit_date = tp_hit_row['date']
                exit_price = tp_hit_row['straddle_price']
                exit_spot_price = tp_hit_row['spot_price']
                exit_reason = 'take_profit'
                days_held = (exit_date - entry_date).days
                
                # Calculate actual return
                profit = entry_price - exit_price
                actual_return_pct = (profit / entry_price) * 100
                
            else:
                # TP was never hit - hold to end
                exit_row = valid_data.iloc[-1]
                exit_date = exit_row['date']
                exit_price = exit_row['straddle_price']
                exit_spot_price = exit_row['spot_price']
                exit_reason = 'held_to_end'
                days_held = (exit_date - entry_date).days
                
                # Calculate actual return
                profit = entry_price - exit_price
                actual_return_pct = (profit / entry_price) * 100
            
            # Calculate spot price movement
            spot_change_pct = ((exit_spot_price - entry_spot_price) / entry_spot_price) * 100
            
            return {
                'filename': filename,
                'symbol': symbol,
                'expiry_date': expiry,
                'strike': strike,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'entry_price': entry_price,
                'entry_spot_price': entry_spot_price,
                'moneyness_category': moneyness_category,
                'strike_distance_pct': distance_pct,
                'tp_target_price': tp_price,
                'tp_target_pct': self.take_profit_pct,
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'exit_price': exit_price,
                'exit_spot_price': exit_spot_price,
                'exit_reason': exit_reason,
                'tp_hit': exit_reason == 'take_profit',
                'days_held': days_held,
                'profit': profit,
                'actual_return_pct': actual_return_pct,
                'spot_change_pct': spot_change_pct,
                'total_data_points': len(valid_data)
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e)
            }
    
    def process_all_files(self):
        """Process all straddle files."""
        self.logger.info(f"Processing all straddle files with {self.take_profit_pct}% take profit by moneyness...")
        
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
                self.logger.info(f"Processed: {result['filename']} - {result['moneyness_category']} - {status} - Return: {result['actual_return_pct']:.2f}%")
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
        output_path = os.path.join(self.output_dir, f'moneyness_tp_{int(self.take_profit_pct)}pct_results_{timestamp}.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary statistics by moneyness category."""
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
        
        # Overall statistics
        total_contracts = len(valid_results)
        
        print(f"\n{'='*80}")
        print(f"SHORT STRADDLE WITH {self.take_profit_pct}% TAKE PROFIT BY MONEYNESS")
        print(f"{'='*80}")
        print(f"Total contracts processed: {total_contracts}")
        print(f"Errors encountered: {len(error_results)}")
        print()
        
        # Moneyness categories breakdown
        moneyness_counts = df['moneyness_category'].value_counts()
        print("MONEYNESS DISTRIBUTION:")
        for category, count in moneyness_counts.items():
            pct = (count / total_contracts) * 100
            print(f"  {category}: {count} contracts ({pct:.1f}%)")
        print()
        
        # Analysis by moneyness category
        categories = df['moneyness_category'].unique()
        
        print("PERFORMANCE BY MONEYNESS CATEGORY:")
        print("="*80)
        
        summary_data = []
        
        for category in sorted(categories):
            cat_df = df[df['moneyness_category'] == category]
            
            if len(cat_df) == 0:
                continue
            
            # Statistics for this category
            cat_count = len(cat_df)
            tp_hit_count = len(cat_df[cat_df['tp_hit'] == True])
            tp_hit_rate = (tp_hit_count / cat_count) * 100
            
            profitable_count = len(cat_df[cat_df['actual_return_pct'] > 0])
            profitable_rate = (profitable_count / cat_count) * 100
            
            avg_return = cat_df['actual_return_pct'].mean()
            median_return = cat_df['actual_return_pct'].median()
            std_return = cat_df['actual_return_pct'].std()
            
            avg_days = cat_df['days_held'].mean()
            avg_distance = cat_df['strike_distance_pct'].mean()
            
            # TP hit vs held statistics
            tp_df = cat_df[cat_df['tp_hit'] == True]
            held_df = cat_df[cat_df['tp_hit'] == False]
            
            avg_days_tp = tp_df['days_held'].mean() if not tp_df.empty else 0
            avg_return_tp = tp_df['actual_return_pct'].mean() if not tp_df.empty else 0
            avg_return_held = held_df['actual_return_pct'].mean() if not held_df.empty else 0
            
            print(f"\n{category} ({cat_count} contracts)")
            print("-" * 40)
            print(f"Average strike distance: {avg_distance:.1f}%")
            print(f"Take profit hit rate: {tp_hit_count}/{cat_count} ({tp_hit_rate:.1f}%)")
            print(f"Profitable rate: {profitable_count}/{cat_count} ({profitable_rate:.1f}%)")
            print(f"Average return: {avg_return:.2f}%")
            print(f"Median return: {median_return:.2f}%")
            print(f"Standard deviation: {std_return:.2f}%")
            print(f"Average days held: {avg_days:.1f}")
            
            if tp_hit_count > 0:
                print(f"  TP hit avg return: {avg_return_tp:.2f}% (avg {avg_days_tp:.1f} days)")
            if len(held_df) > 0:
                print(f"  Held to end avg return: {avg_return_held:.2f}%")
            
            # Store for comparison table
            summary_data.append({
                'Category': category,
                'Count': cat_count,
                'TP Hit Rate': f"{tp_hit_rate:.1f}%",
                'Profitable Rate': f"{profitable_rate:.1f}%",
                'Avg Return': f"{avg_return:.2f}%",
                'Avg Days': f"{avg_days:.1f}",
                'Avg Distance': f"{avg_distance:.1f}%"
            })
        
        # Comparison table
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Category':<12} {'Count':<6} {'TP Hit':<8} {'Profit':<8} {'Avg Ret':<8} {'Days':<6} {'Distance':<8}")
        print("-" * 80)
        for data in summary_data:
            print(f"{data['Category']:<12} {data['Count']:<6} {data['TP Hit Rate']:<8} {data['Profitable Rate']:<8} "
                  f"{data['Avg Return']:<8} {data['Avg Days']:<6} {data['Avg Distance']:<8}")
        
        # Best performing category
        if summary_data:
            best_category = max(summary_data, key=lambda x: float(x['Avg Return'].replace('%', '')))
            worst_category = min(summary_data, key=lambda x: float(x['Avg Return'].replace('%', '')))
            
            print(f"\nKEY INSIGHTS:")
            print(f"Best performing moneyness: {best_category['Category']} ({best_category['Avg Return']} avg return)")
            print(f"Worst performing moneyness: {worst_category['Category']} ({worst_category['Avg Return']} avg return)")
        
        print(f"\n{'='*80}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("="*80)
        print(f"MONEYNESS-BASED SHORT STRADDLE WITH {self.take_profit_pct}% TAKE PROFIT")
        print("="*80)
        print(f"Strategy: Enter short straddle on first day, exit at {self.take_profit_pct}% profit or hold to end")
        print("Analysis separated by call leg moneyness:")
        print("- ATM: Call strike within 2.5% of spot")
        print("- 5%_OTM: Call strike ~5% above spot") 
        print("- 10%_OTM: Call strike ~10% above spot")
        print("- 15%_OTM: Call strike ~15% above spot")
        print("- 20%_OTM: Call strike ~20% above spot")
        print("- 25%_OTM: Call strike ~25% above spot")
        print("- 30%_OTM: Call strike ~30% above spot")
        print("- ITM categories: Call strike below spot")
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
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/moneyness_tp_analysis"
    take_profit_pct = 50.0  # 50% take profit
    
    # Create analyzer and run
    analyzer = MoneynessTakeProfitAnalyzer(
        straddle_data_dir=straddle_data_dir,
        output_dir=output_dir,
        take_profit_pct=take_profit_pct
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()