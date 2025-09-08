import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

class StraddleMoneynessAnalyzer:
    def __init__(self, straddle_data_dir: str, output_dir: str = None):
        """
        Initialize the straddle moneyness analyzer.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            output_dir: Directory to save results
        """
        self.straddle_data_dir = straddle_data_dir
        self.output_dir = output_dir or f"{straddle_data_dir}_moneyness_analysis"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.results = []
    
    def categorize_moneyness(self, spot_price: float, strike_price: float) -> str:
        """
        Categorize straddle by moneyness using call-like logic.
        
        Args:
            spot_price: Current spot price
            strike_price: Strike price of the straddle
            
        Returns:
            String representing moneyness category
        """
        if spot_price <= 0:
            return 'UNKNOWN'
        
        # Calculate moneyness as strike/spot (same as call moneyness)
        moneyness_ratio = strike_price / spot_price
        
        # Calculate percentage difference
        distance_pct = ((strike_price - spot_price) / spot_price) * 100
        
        # Categorize based on distance
        if abs(distance_pct) <= 2.5:
            return 'ATM'  # At-the-money: within 2.5%
        elif 2.5 < distance_pct <= 7.5:
            return '5%_OTM'  # ~5% out-of-the-money
        elif 7.5 < distance_pct <= 12.5:
            return '10%_OTM'  # ~10% out-of-the-money
        elif 12.5 < distance_pct <= 17.5:
            return '15%_OTM'  # ~15% out-of-the-money
        elif 17.5 < distance_pct <= 22.5:
            return '20%_OTM'  # ~20% out-of-the-money
        elif 22.5 < distance_pct <= 27.5:
            return '25%_OTM'  # ~25% out-of-the-money
        elif 27.5 < distance_pct <= 32.5:
            return '30%_OTM'  # ~30% out-of-the-money
        elif distance_pct > 32.5:
            return 'FAR_OTM'  # Far out-of-the-money
        elif -7.5 <= distance_pct < -2.5:
            return '5%_ITM'  # ~5% in-the-money
        elif -12.5 <= distance_pct < -7.5:
            return '10%_ITM'  # ~10% in-the-money
        elif -17.5 <= distance_pct < -12.5:
            return '15%_ITM'  # ~15% in-the-money
        elif distance_pct < -17.5:
            return 'FAR_ITM'  # Far in-the-money
        else:
            return 'OTHER'
    
    def analyze_straddle(self, filepath: str) -> dict:
        """Analyze a single straddle file for moneyness and returns."""
        filename = os.path.basename(filepath)
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get valid price and spot price data
            valid_data = df.dropna(subset=['straddle_price', 'spot_price']).copy()
            
            if len(valid_data) < 2:
                return {
                    'filename': filename,
                    'error': 'Insufficient valid data'
                }
            
            # Get first and last data points
            first_row = valid_data.iloc[0]
            last_row = valid_data.iloc[-1]
            
            first_price = first_row['straddle_price']
            last_price = last_row['straddle_price']
            first_date = first_row['date']
            last_date = last_row['date']
            first_spot_price = first_row['spot_price']
            last_spot_price = last_row['spot_price']
            
            # Parse contract info from filename
            # Expected format: SYMBOL_YYYY-MM-DD_STRIKE_STRADDLE.csv
            parts = filename.replace('_STRADDLE.csv', '').split('_')
            symbol = parts[0] if len(parts) > 0 else 'UNKNOWN'
            expiry = parts[1] if len(parts) > 1 else 'UNKNOWN'
            strike = float(parts[2]) if len(parts) > 2 else 0
            
            # Calculate moneyness using first spot price
            moneyness_category = self.categorize_moneyness(first_spot_price, strike)
            distance_pct = ((strike - first_spot_price) / first_spot_price) * 100
            moneyness_ratio = strike / first_spot_price
            
            # Calculate short straddle return
            # Short position: Sell at first_price, buy back at last_price
            # Profit = first_price - last_price
            # Return = (first_price - last_price) / first_price * 100
            profit = first_price - last_price
            return_pct = (profit / first_price) * 100 if first_price > 0 else 0
            
            # Calculate days held
            days_held = (last_date - first_date).days
            
            # Calculate spot price movement
            spot_change_pct = ((last_spot_price - first_spot_price) / first_spot_price) * 100
            
            # Calculate some additional metrics
            max_price = valid_data['straddle_price'].max()
            min_price = valid_data['straddle_price'].min()
            max_drawdown_pct = ((max_price - first_price) / first_price) * 100 if first_price > 0 else 0
            max_profit_pct = ((first_price - min_price) / first_price) * 100 if first_price > 0 else 0
            
            return {
                'filename': filename,
                'symbol': symbol,
                'expiry_date': expiry,
                'strike': strike,
                'moneyness_category': moneyness_category,
                'strike_distance_pct': distance_pct,
                'moneyness_ratio': moneyness_ratio,
                'first_date': first_date.strftime('%Y-%m-%d'),
                'last_date': last_date.strftime('%Y-%m-%d'),
                'days_held': days_held,
                'first_spot_price': first_spot_price,
                'last_spot_price': last_spot_price,
                'spot_change_pct': spot_change_pct,
                'first_straddle_price': first_price,
                'last_straddle_price': last_price,
                'max_straddle_price': max_price,
                'min_straddle_price': min_price,
                'profit': profit,
                'return_pct': return_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'max_profit_pct': max_profit_pct,
                'total_data_points': len(valid_data),
                'profitable': return_pct > 0
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e)
            }
    
    def process_all_files(self):
        """Process all straddle files."""
        self.logger.info("Processing all straddle files for moneyness analysis...")
        
        # Find all straddle files
        pattern = os.path.join(self.straddle_data_dir, "*_STRADDLE.csv")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.error(f"No straddle files found in {self.straddle_data_dir}")
            return
        
        self.logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        for filepath in files:
            result = self.analyze_straddle(filepath)
            self.results.append(result)
            
            if 'error' not in result:
                self.logger.info(f"Processed: {result['symbol']} {result['moneyness_category']} - Return: {result['return_pct']:.2f}%")
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
        output_path = os.path.join(self.output_dir, f'straddle_moneyness_results_{timestamp}.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary_by_moneyness(self):
        """Print summary statistics grouped by moneyness."""
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
        print("SHORT STRADDLE ANALYSIS BY MONEYNESS")
        print(f"{'='*80}")
        print(f"Total contracts processed: {total_contracts}")
        print(f"Errors encountered: {len(error_results)}")
        print()
        
        # Moneyness distribution
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
            profitable_count = len(cat_df[cat_df['profitable'] == True])
            profitable_rate = (profitable_count / cat_count) * 100
            
            avg_return = cat_df['return_pct'].mean()
            median_return = cat_df['return_pct'].median()
            std_return = cat_df['return_pct'].std()
            min_return = cat_df['return_pct'].min()
            max_return = cat_df['return_pct'].max()
            
            avg_days = cat_df['days_held'].mean()
            avg_distance = cat_df['strike_distance_pct'].mean()
            avg_spot_change = cat_df['spot_change_pct'].mean()
            
            print(f"\n{category} ({cat_count} contracts)")
            print("-" * 40)
            print(f"Average strike distance: {avg_distance:+.1f}%")
            print(f"Profitable rate: {profitable_count}/{cat_count} ({profitable_rate:.1f}%)")
            print(f"Average return: {avg_return:+.2f}%")
            print(f"Median return: {median_return:+.2f}%")
            print(f"Return range: {min_return:+.2f}% to {max_return:+.2f}%")
            print(f"Standard deviation: {std_return:.2f}%")
            print(f"Average days held: {avg_days:.1f}")
            print(f"Average spot price change: {avg_spot_change:+.2f}%")
            
            # Store for comparison table
            summary_data.append({
                'Category': category,
                'Count': cat_count,
                'Profitable Rate': f"{profitable_rate:.1f}%",
                'Avg Return': f"{avg_return:+.2f}%",
                'Median Return': f"{median_return:+.2f}%",
                'Std Dev': f"{std_return:.2f}%",
                'Avg Days': f"{avg_days:.1f}",
                'Avg Distance': f"{avg_distance:+.1f}%"
            })
        
        # Comparison table
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Category':<12} {'Count':<6} {'Profit%':<8} {'Avg Ret':<9} {'Med Ret':<9} {'Std Dev':<8} {'Days':<6} {'Distance':<9}")
        print("-" * 80)
        for data in summary_data:
            print(f"{data['Category']:<12} {data['Count']:<6} {data['Profitable Rate']:<8} "
                  f"{data['Avg Return']:<9} {data['Median Return']:<9} {data['Std Dev']:<8} "
                  f"{data['Avg Days']:<6} {data['Avg Distance']:<9}")
        
        # Best and worst performing categories
        if summary_data:
            best_category = max(summary_data, key=lambda x: float(x['Avg Return'].replace('%', '').replace('+', '')))
            worst_category = min(summary_data, key=lambda x: float(x['Avg Return'].replace('%', '').replace('+', '')))
            
            print(f"\nKEY INSIGHTS:")
            print(f"Best performing moneyness: {best_category['Category']} ({best_category['Avg Return']} avg return)")
            print(f"Worst performing moneyness: {worst_category['Category']} ({worst_category['Avg Return']} avg return)")
            
            # Overall profitability
            overall_profitable_rate = (len(df[df['profitable'] == True]) / total_contracts) * 100
            overall_avg_return = df['return_pct'].mean()
            
            print(f"\nOVERALL STATISTICS:")
            print(f"Total profitable rate: {overall_profitable_rate:.1f}%")
            print(f"Overall average return: {overall_avg_return:+.2f}%")
        
        print(f"\n{'='*80}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("="*80)
        print("STRADDLE MONEYNESS GROUPING AND SHORT RETURNS ANALYSIS")
        print("="*80)
        print("Analyzing short straddle returns grouped by moneyness categories:")
        print("- ATM: Strike within 2.5% of spot")
        print("- 5%_OTM: Strike ~5% above spot") 
        print("- 10%_OTM: Strike ~10% above spot")
        print("- 15%_OTM: Strike ~15% above spot")
        print("- 20%_OTM: Strike ~20% above spot")
        print("- 25%_OTM: Strike ~25% above spot")
        print("- 30%_OTM: Strike ~30% above spot")
        print("- ITM categories: Strike below spot")
        print()
        print("Returns calculated as: (First Price - Last Price) / First Price * 100")
        print("(Positive returns = profitable short straddle)")
        print(f"Input directory: {self.straddle_data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Process files
        self.process_all_files()
        
        # Save results
        output_path = self.save_results()
        
        # Print summary
        self.print_summary_by_moneyness()
        
        if output_path:
            print(f"\nDetailed results saved to: {output_path}")

def main():
    # Configuration
    straddle_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_moneyness_analysis"
    
    # Create analyzer and run
    analyzer = StraddleMoneynessAnalyzer(
        straddle_data_dir=straddle_data_dir,
        output_dir=output_dir
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()