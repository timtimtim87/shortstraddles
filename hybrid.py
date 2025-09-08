import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

class OTMBreachAnalyzer:
    def __init__(self, straddle_data_dir: str, output_dir: str = None, min_otm_pct: float = 10.0):
        """
        Initialize the OTM breach analyzer.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            output_dir: Directory to save results
            min_otm_pct: Minimum OTM percentage to include (default 10%)
        """
        self.straddle_data_dir = straddle_data_dir
        self.output_dir = output_dir or f"{straddle_data_dir}_otm_breach"
        self.min_otm_pct = min_otm_pct
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.results = []
    
    def calculate_otm_percentage(self, spot_price: float, strike_price: float) -> float:
        """Calculate how far OTM the strike is from spot."""
        if spot_price <= 0:
            return 0
        return abs((strike_price - spot_price) / spot_price) * 100
    
    def check_strike_breach(self, df: pd.DataFrame, strike: float) -> dict:
        """
        Check if spot price ever reached the strike price.
        
        Args:
            df: DataFrame with date, spot_price, straddle_price columns
            strike: Strike price to check
            
        Returns:
            Dictionary with breach information
        """
        # Sort by date
        df = df.sort_values('date').copy()
        
        # Find where spot price reached or crossed the strike
        # For straddles, breach happens when spot reaches strike from either direction
        breach_mask = (df['spot_price'] >= strike) | (df['spot_price'] <= strike)
        
        # More specifically, find significant moves toward the strike
        initial_spot = df['spot_price'].iloc[0]
        
        if strike > initial_spot:
            # Strike is above initial spot - breach when spot reaches or exceeds strike
            breach_points = df[df['spot_price'] >= strike]
        else:
            # Strike is below initial spot - breach when spot reaches or falls to strike
            breach_points = df[df['spot_price'] <= strike]
        
        if breach_points.empty:
            # No breach occurred
            return {
                'breached': False,
                'final_spot': df['spot_price'].iloc[-1],
                'final_straddle_price': df['straddle_price'].iloc[-1],
                'final_date': df['date'].iloc[-1]
            }
        else:
            # Breach occurred - get first occurrence
            first_breach = breach_points.iloc[0]
            return {
                'breached': True,
                'breach_date': first_breach['date'],
                'breach_spot': first_breach['spot_price'],
                'breach_straddle_price': first_breach['straddle_price']
            }
    
    def analyze_straddle_file(self, filepath: str) -> dict:
        """Analyze a single straddle file."""
        filename = os.path.basename(filepath)
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get clean data
            clean_df = df.dropna(subset=['straddle_price', 'spot_price']).copy()
            
            if len(clean_df) < 2:
                return {
                    'filename': filename,
                    'error': 'Not enough data points'
                }
            
            # Parse filename to get contract details
            # Expected: SYMBOL_YYYY-MM-DD_STRIKE_STRADDLE.csv
            base_name = filename.replace('_STRADDLE.csv', '')
            parts = base_name.split('_')
            
            if len(parts) < 3:
                return {
                    'filename': filename,
                    'error': 'Cannot parse filename'
                }
            
            symbol = parts[0]
            expiry_date = parts[1]
            strike = float(parts[2])
            
            # Get initial values
            first_row = clean_df.iloc[0]
            initial_date = first_row['date']
            initial_spot = first_row['spot_price']
            initial_straddle_price = first_row['straddle_price']
            
            # Calculate OTM percentage
            otm_pct = self.calculate_otm_percentage(initial_spot, strike)
            
            # Skip if not OTM enough
            if otm_pct < self.min_otm_pct:
                return {
                    'filename': filename,
                    'symbol': symbol,
                    'skipped': True,
                    'reason': f'Only {otm_pct:.1f}% OTM (need {self.min_otm_pct}%+)'
                }
            
            # Check for strike breach
            breach_info = self.check_strike_breach(clean_df, strike)
            
            # Calculate final return (entry to end)
            final_row = clean_df.iloc[-1]
            final_return = ((initial_straddle_price - final_row['straddle_price']) / initial_straddle_price) * 100
            
            # Build result
            result = {
                'filename': filename,
                'symbol': symbol,
                'expiry_date': expiry_date,
                'strike': strike,
                'initial_date': initial_date.strftime('%Y-%m-%d'),
                'initial_spot_price': initial_spot,
                'initial_straddle_price': initial_straddle_price,
                'otm_percentage': otm_pct,
                'strike_direction': 'above' if strike > initial_spot else 'below',
                'total_days': (final_row['date'] - initial_date).days,
                'final_return_pct': final_return,
                'skipped': False
            }
            
            if breach_info['breached']:
                # Strike was reached
                days_to_breach = (breach_info['breach_date'] - initial_date).days
                breach_return = ((initial_straddle_price - breach_info['breach_straddle_price']) / initial_straddle_price) * 100
                spot_move_pct = ((breach_info['breach_spot'] - initial_spot) / initial_spot) * 100
                
                result.update({
                    'strike_reached': True,
                    'days_to_breach': days_to_breach,
                    'breach_date': breach_info['breach_date'].strftime('%Y-%m-%d'),
                    'breach_spot_price': breach_info['breach_spot'],
                    'breach_straddle_price': breach_info['breach_straddle_price'],
                    'return_at_breach_pct': breach_return,
                    'spot_move_to_breach_pct': spot_move_pct
                })
            else:
                # Strike was never reached
                result.update({
                    'strike_reached': False,
                    'days_to_breach': None,
                    'breach_date': None,
                    'breach_spot_price': None,
                    'breach_straddle_price': None,
                    'return_at_breach_pct': None,
                    'spot_move_to_breach_pct': None
                })
            
            return result
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e)
            }
    
    def process_all_files(self):
        """Process all straddle files in the directory."""
        self.logger.info(f"Starting OTM breach analysis (min {self.min_otm_pct}% OTM)...")
        
        # Find straddle files
        pattern = os.path.join(self.straddle_data_dir, "*_STRADDLE.csv")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.error(f"No straddle files found in {self.straddle_data_dir}")
            return
        
        self.logger.info(f"Found {len(files)} straddle files")
        
        # Process each file
        for filepath in files:
            result = self.analyze_straddle_file(filepath)
            self.results.append(result)
            
            # Log progress
            if 'error' in result:
                self.logger.error(f"Error: {result['filename']} - {result['error']}")
            elif result.get('skipped', False):
                self.logger.info(f"Skipped: {result['symbol']} - {result['reason']}")
            else:
                breach_status = "BREACHED" if result['strike_reached'] else "NOT BREACHED"
                self.logger.info(f"Analyzed: {result['symbol']} ({result['otm_percentage']:.1f}% OTM) - {breach_status}")
        
        self.logger.info(f"Completed processing {len(self.results)} files")
    
    def save_results(self):
        """Save results to CSV files."""
        if not self.results:
            self.logger.warning("No results to save")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        all_df = pd.DataFrame(self.results)
        all_path = os.path.join(self.output_dir, f'otm_breach_all_results_{timestamp}.csv')
        all_df.to_csv(all_path, index=False)
        self.logger.info(f"All results saved: {all_path}")
        
        # Save only analyzed results (exclude skipped and errors)
        analyzed_df = all_df[
            (~all_df.get('skipped', False)) & 
            (~all_df['symbol'].isna()) & 
            (~all_df.get('error', pd.Series(dtype='object')).notna())
        ]
        
        if not analyzed_df.empty:
            analyzed_path = os.path.join(self.output_dir, f'otm_breach_analyzed_{timestamp}.csv')
            analyzed_df.to_csv(analyzed_path, index=False)
            self.logger.info(f"Analyzed results saved: {analyzed_path}")
            return analyzed_path
        
        return all_path
    
    def print_summary(self):
        """Print comprehensive summary."""
        if not self.results:
            print("No results to summarize")
            return
        
        # Separate results by type
        analyzed = [r for r in self.results if not r.get('skipped', False) and 'error' not in r and not pd.isna(r.get('symbol'))]
        skipped = [r for r in self.results if r.get('skipped', False)]
        errors = [r for r in self.results if 'error' in r]
        
        print(f"\n{'='*80}")
        print(f"OTM STRADDLE STRIKE BREACH ANALYSIS (MIN {self.min_otm_pct}% OTM)")
        print(f"{'='*80}")
        print(f"Total files processed: {len(self.results)}")
        print(f"Analyzed (qualifying OTM): {len(analyzed)}")
        print(f"Skipped (not OTM enough): {len(skipped)}")
        print(f"Errors: {len(errors)}")
        print()
        
        if not analyzed:
            print("No qualifying OTM straddles found for analysis")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(analyzed)
        
        # Basic breach statistics
        total_analyzed = len(df)
        breached = df[df['strike_reached'] == True]
        not_breached = df[df['strike_reached'] == False]
        
        breach_count = len(breached)
        breach_rate = (breach_count / total_analyzed) * 100
        
        print("STRIKE BREACH OVERVIEW:")
        print(f"Total OTM straddles analyzed: {total_analyzed}")
        print(f"Strikes reached: {breach_count} ({breach_rate:.1f}%)")
        print(f"Strikes not reached: {len(not_breached)} ({100-breach_rate:.1f}%)")
        print()
        
        # Analysis when strikes were breached
        if breach_count > 0:
            print("WHEN STRIKES WERE REACHED:")
            print(f"Average days to breach: {breached['days_to_breach'].mean():.1f}")
            print(f"Median days to breach: {breached['days_to_breach'].median():.1f}")
            print(f"Fastest breach: {breached['days_to_breach'].min()} days")
            print(f"Slowest breach: {breached['days_to_breach'].max()} days")
            print()
            
            print("RETURNS WHEN STRIKE REACHED:")
            print(f"Average return at breach: {breached['return_at_breach_pct'].mean():.2f}%")
            print(f"Median return at breach: {breached['return_at_breach_pct'].median():.2f}%")
            print(f"Best return at breach: {breached['return_at_breach_pct'].max():.2f}%")
            print(f"Worst return at breach: {breached['return_at_breach_pct'].min():.2f}%")
            print()
        
        # Analysis when strikes were not breached
        if len(not_breached) > 0:
            print("WHEN STRIKES WERE NOT REACHED:")
            print(f"Average final return: {not_breached['final_return_pct'].mean():.2f}%")
            print(f"Median final return: {not_breached['final_return_pct'].median():.2f}%")
            print(f"Best final return: {not_breached['final_return_pct'].max():.2f}%")
            print(f"Worst final return: {not_breached['final_return_pct'].min():.2f}%")
            print()
        
        # STRATEGY COMPARISON: Exit on breach vs Hold to expiry
        print("="*80)
        print("STRATEGY COMPARISON: EXIT ON BREACH vs HOLD TO EXPIRY")
        print("="*80)
        
        if breach_count > 0 and len(not_breached) > 0:
            # Strategy 1: Exit when strike is breached
            breach_avg_return = breached['return_at_breach_pct'].mean()
            breach_profitable_rate = (len(breached[breached['return_at_breach_pct'] > 0]) / len(breached)) * 100
            
            # Strategy 2: Hold to expiry (only non-breached contribute)
            hold_avg_return = not_breached['final_return_pct'].mean()
            hold_profitable_rate = (len(not_breached[not_breached['final_return_pct'] > 0]) / len(not_breached)) * 100
            
            # Combined strategy performance
            total_breach_return = breached['return_at_breach_pct'].sum()
            total_hold_return = not_breached['final_return_pct'].sum()
            combined_avg_return = (total_breach_return + total_hold_return) / total_analyzed
            
            print(f"STRATEGY 1 - Exit when strike breached ({breach_count} cases):")
            print(f"  Average return: {breach_avg_return:.2f}%")
            print(f"  Profitable rate: {breach_profitable_rate:.1f}%")
            
            print(f"\nSTRATEGY 2 - Hold to expiry ({len(not_breached)} cases):")
            print(f"  Average return: {hold_avg_return:.2f}%")
            print(f"  Profitable rate: {hold_profitable_rate:.1f}%")
            
            print(f"\nCOMBINED STRATEGY (Exit on breach OR hold to expiry):")
            print(f"  Overall average return: {combined_avg_return:.2f}%")
            print(f"  Total profitable trades: {len(breached[breached['return_at_breach_pct'] > 0]) + len(not_breached[not_breached['final_return_pct'] > 0])}/{total_analyzed}")
            
            # Risk analysis
            breach_worst = breached['return_at_breach_pct'].min()
            hold_worst = not_breached['final_return_pct'].min()
            
            print(f"\nRISK ANALYSIS:")
            print(f"  Worst return when breached: {breach_worst:.2f}%")
            print(f"  Worst return when held: {hold_worst:.2f}%")
            print(f"  Overall worst case: {min(breach_worst, hold_worst):.2f}%")
            
        elif len(not_breached) > 0:
            print(f"Only non-breached data available:")
            print(f"Average return (hold to expiry): {not_breached['final_return_pct'].mean():.2f}%")
            
        print()
        
        # OTM percentage analysis
        print("ANALYSIS BY OTM PERCENTAGE:")
        print(f"{'OTM Range':<12} {'Count':<6} {'Breached':<9} {'Breach %':<9} {'Avg Return@Breach':<16} {'Avg Final Return':<15}")
        print("-" * 80)
        
        # Create OTM buckets
        otm_bins = [10, 15, 20, 25, 30, 50, 100]
        otm_labels = ['10-15%', '15-20%', '20-25%', '25-30%', '30-50%', '50%+']
        
        df['otm_bucket'] = pd.cut(df['otm_percentage'], bins=otm_bins, labels=otm_labels, include_lowest=True)
        
        for bucket in otm_labels:
            bucket_df = df[df['otm_bucket'] == bucket]
            if len(bucket_df) == 0:
                continue
                
            bucket_breached = bucket_df[bucket_df['strike_reached'] == True]
            
            count = len(bucket_df)
            breached_count = len(bucket_breached)
            breach_pct = (breached_count / count) * 100 if count > 0 else 0
            
            avg_breach_return = bucket_breached['return_at_breach_pct'].mean() if breached_count > 0 else None
            avg_final_return = bucket_df['final_return_pct'].mean()
            
            breach_return_str = f"{avg_breach_return:.1f}%" if avg_breach_return is not None else "N/A"
            
            print(f"{bucket:<12} {count:<6} {breached_count:<9} {breach_pct:<8.1f}% {breach_return_str:<16} {avg_final_return:<14.1f}%")
        
        print()
        
        # Show some examples
        if breach_count > 0:
            print("FASTEST BREACHES:")
            fastest = breached.nsmallest(3, 'days_to_breach')[['symbol', 'otm_percentage', 'days_to_breach', 'return_at_breach_pct']]
            for _, row in fastest.iterrows():
                print(f"  {row['symbol']}: {row['otm_percentage']:.1f}% OTM, breached in {row['days_to_breach']} days, {row['return_at_breach_pct']:.2f}% return")
            
            print()
            print("BEST RETURNS AT BREACH:")
            best_returns = breached.nlargest(3, 'return_at_breach_pct')[['symbol', 'otm_percentage', 'days_to_breach', 'return_at_breach_pct']]
            for _, row in best_returns.iterrows():
                print(f"  {row['symbol']}: {row['otm_percentage']:.1f}% OTM, {row['days_to_breach']} days, {row['return_at_breach_pct']:.2f}% return")
        
        print(f"\n{'='*80}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("="*80)
        print(f"OTM STRADDLE STRIKE BREACH ANALYZER")
        print("="*80)
        print(f"Analyzing straddles that are {self.min_otm_pct}%+ away from spot price")
        print("Questions answered:")
        print("1. Did the spot price ever reach the strike?")
        print("2. If yes, what was the return at that moment?")
        print("3. How many days did it take?")
        print()
        print("Note: Returns calculated for SHORT straddle positions")
        print("(Positive = profit from short position)")
        print(f"Data directory: {self.straddle_data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Run the analysis
        self.process_all_files()
        
        # Save results
        output_path = self.save_results()
        
        # Print summary
        self.print_summary()
        
        print(f"\nResults saved to: {output_path}")

def main():
    # Configuration
    straddle_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/otm_breach_analysis"
    min_otm_pct = 10.0  # Analyze straddles 10%+ OTM
    
    # Create and run analyzer
    analyzer = OTMBreachAnalyzer(
        straddle_data_dir=straddle_data_dir,
        output_dir=output_dir,
        min_otm_pct=min_otm_pct
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()