import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import logging
import json

class ShortStraddleAnalyzer:
    def __init__(self, straddle_data_dir: str, spot_prices_path: str, output_dir: str = None):
        """
        Initialize the short straddle analyzer.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            spot_prices_path: Path to the spot prices CSV file
            output_dir: Directory to save analysis results
        """
        self.straddle_data_dir = straddle_data_dir
        self.spot_prices_path = spot_prices_path
        self.output_dir = output_dir or f"{straddle_data_dir}_analysis"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load spot prices
        self.load_spot_prices()
        
        self.analysis_results = []
    
    def load_spot_prices(self):
        """Load spot prices data."""
        try:
            self.spot_prices_df = pd.read_csv(self.spot_prices_path)
            self.spot_prices_df['Date'] = pd.to_datetime(self.spot_prices_df['Date'])
            
            # Create a lookup dictionary for faster access
            self.spot_price_lookup = {}
            for _, row in self.spot_prices_df.iterrows():
                key = (row['Ticker'], row['Date'].strftime('%Y-%m-%d'))
                self.spot_price_lookup[key] = row['Close']
            
            self.logger.info(f"Loaded {len(self.spot_prices_df)} spot price records")
            
        except Exception as e:
            self.logger.error(f"Error loading spot prices: {e}")
            raise
    
    def get_spot_price(self, symbol: str, date: str) -> float:
        """Get spot price for a given symbol and date."""
        key = (symbol, date)
        spot_price = self.spot_price_lookup.get(key)
        
        if spot_price is None:
            # Try to find the closest date within a reasonable range
            date_dt = pd.to_datetime(date)
            symbol_data = self.spot_prices_df[self.spot_prices_df['Ticker'] == symbol].copy()
            
            if not symbol_data.empty:
                symbol_data['date_diff'] = abs(symbol_data['Date'] - date_dt)
                closest = symbol_data.loc[symbol_data['date_diff'].idxmin()]
                
                if closest['date_diff'].days <= 3:  # Within 3 days
                    return closest['Close']
        
        return spot_price
    
    def load_straddle_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare straddle data."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def parse_straddle_info(self, filename: str) -> dict:
        """Extract contract information from straddle filename."""
        try:
            # Example: AAPL_2025-01-17_210_STRADDLE.csv
            basename = filename.replace('_STRADDLE.csv', '')
            parts = basename.split('_')
            
            return {
                'symbol': parts[0],
                'expiry_date': parts[1],
                'strike': float(parts[2])
            }
        except Exception as e:
            self.logger.error(f"Error parsing filename {filename}: {e}")
            return None
    
    def analyze_price_movement(self, symbol: str, start_date: str, end_date: str, strike: float) -> dict:
        """
        Analyze if and when the spot price reached the strike price.
        
        Uses actual spot prices data, not the straddle file spot prices.
        """
        try:
            # Get all spot prices for this symbol in the date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            symbol_data = self.spot_prices_df[
                (self.spot_prices_df['Ticker'] == symbol) &
                (self.spot_prices_df['Date'] >= start_dt) &
                (self.spot_prices_df['Date'] <= end_dt)
            ].copy().sort_values('Date')
            
            if symbol_data.empty:
                return {
                    'error': 'No spot price data found for date range',
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            initial_price = symbol_data['Close'].iloc[0]
            initial_date = symbol_data['Date'].iloc[0]
            
            # Check if price ever reached the strike
            # For straddles, breach occurs when price moves AT OR BEYOND the strike
            if strike > initial_price:
                # Strike is above initial price - breach when price reaches or exceeds strike
                breach_condition = symbol_data['Close'] >= strike
                direction = 'upward'
            else:
                # Strike is below initial price - breach when price reaches or falls below strike
                breach_condition = symbol_data['Close'] <= strike
                direction = 'downward'
            
            breaches = symbol_data[breach_condition]
            
            if breaches.empty:
                # No breach occurred
                final_price = symbol_data['Close'].iloc[-1]
                final_date = symbol_data['Date'].iloc[-1]
                
                return {
                    'breached': False,
                    'initial_price': initial_price,
                    'initial_date': initial_date.strftime('%Y-%m-%d'),
                    'final_price': final_price,
                    'final_date': final_date.strftime('%Y-%m-%d'),
                    'strike': strike,
                    'direction_needed': direction,
                    'max_price': symbol_data['Close'].max(),
                    'min_price': symbol_data['Close'].min(),
                    'price_change_pct': ((final_price - initial_price) / initial_price) * 100
                }
            else:
                # Breach occurred
                first_breach = breaches.iloc[0]
                days_to_breach = (first_breach['Date'] - initial_date).days
                
                return {
                    'breached': True,
                    'initial_price': initial_price,
                    'initial_date': initial_date.strftime('%Y-%m-%d'),
                    'breach_price': first_breach['Close'],
                    'breach_date': first_breach['Date'].strftime('%Y-%m-%d'),
                    'days_to_breach': days_to_breach,
                    'strike': strike,
                    'direction': direction,
                    'price_change_to_breach_pct': ((first_breach['Close'] - initial_price) / initial_price) * 100
                }
                
        except Exception as e:
            return {
                'error': f'Analysis error: {str(e)}',
                'symbol': symbol
            }
    
    def calculate_short_straddle_return(self, df: pd.DataFrame, price_analysis: dict) -> dict:
        """
        Calculate short straddle returns based on price movement analysis.
        """
        try:
            valid_data = df.dropna(subset=['straddle_price']).copy()
            
            if valid_data.empty:
                return {'error': 'No valid straddle price data'}
            
            initial_price = valid_data['straddle_price'].iloc[0]
            initial_date = valid_data['date'].iloc[0]
            
            if initial_price <= 0:
                return {'error': 'Invalid initial straddle price'}
            
            if price_analysis.get('breached', False):
                # Find straddle price on or near breach date
                breach_date = pd.to_datetime(price_analysis['breach_date'])
                
                # Find closest straddle data to breach date
                valid_data['date_diff'] = abs(valid_data['date'] - breach_date)
                closest_breach_idx = valid_data['date_diff'].idxmin()
                closest_breach_data = valid_data.loc[closest_breach_idx]
                
                exit_price = closest_breach_data['straddle_price']
                exit_date = closest_breach_data['date']
                exit_reason = 'strike_breached'
                
            else:
                # Hold to end of data (expiration or data cutoff)
                exit_price = valid_data['straddle_price'].iloc[-1]
                exit_date = valid_data['date'].iloc[-1]
                exit_reason = 'held_to_end'
            
            days_held = (exit_date - initial_date).days
            
            # Calculate short straddle return
            # For short positions: Profit = Premium Received - Cost to Buy Back
            profit = initial_price - exit_price
            return_pct = (profit / initial_price) * 100
            
            return {
                'initial_straddle_price': initial_price,
                'initial_date': initial_date.strftime('%Y-%m-%d'),
                'exit_straddle_price': exit_price,
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'exit_reason': exit_reason,
                'days_held': days_held,
                'profit': profit,
                'return_pct': return_pct,
                'annualized_return': (return_pct * 365 / days_held) if days_held > 0 else 0
            }
            
        except Exception as e:
            return {'error': f'Return calculation error: {str(e)}'}
    
    def analyze_straddle(self, filepath: str) -> dict:
        """Analyze a single straddle contract."""
        filename = os.path.basename(filepath)
        self.logger.info(f"Analyzing: {filename}")
        
        # Load straddle data
        df = self.load_straddle_data(filepath)
        if df is None or df.empty:
            return {'error': 'Could not load straddle data', 'filename': filename}
        
        # Parse contract info
        contract_info = self.parse_straddle_info(filename)
        if contract_info is None:
            return {'error': 'Could not parse contract info', 'filename': filename}
        
        # Get date range from straddle data
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
        
        # Analyze price movement using spot prices data
        price_analysis = self.analyze_price_movement(
            contract_info['symbol'], 
            start_date, 
            end_date, 
            contract_info['strike']
        )
        
        if 'error' in price_analysis:
            return {
                'error': price_analysis['error'],
                'filename': filename,
                **contract_info
            }
        
        # Calculate straddle returns
        return_analysis = self.calculate_short_straddle_return(df, price_analysis)
        
        # Combine all analysis
        result = {
            'filename': filename,
            'symbol': contract_info['symbol'],
            'expiry_date': contract_info['expiry_date'],
            'strike': contract_info['strike'],
            'data_start_date': start_date,
            'data_end_date': end_date,
            'total_straddle_data_points': len(df),
            **price_analysis,
            **return_analysis
        }
        
        return result
    
    def analyze_all_straddles(self):
        """Analyze all straddle files."""
        self.logger.info("Starting comprehensive straddle analysis...")
        
        # Find all straddle files
        pattern = os.path.join(self.straddle_data_dir, "*_STRADDLE.csv")
        straddle_files = glob.glob(pattern)
        
        if not straddle_files:
            self.logger.error(f"No straddle files found in {self.straddle_data_dir}")
            return
        
        self.logger.info(f"Found {len(straddle_files)} straddle files to analyze")
        
        # Analyze each file
        for filepath in straddle_files:
            try:
                result = self.analyze_straddle(filepath)
                self.analysis_results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing {filepath}: {e}")
                self.analysis_results.append({
                    'error': str(e),
                    'filename': os.path.basename(filepath)
                })
        
        self.logger.info(f"Analysis complete. Processed {len(self.analysis_results)} contracts")
    
    def create_summary_statistics(self):
        """Create summary statistics from analysis results."""
        # Filter out errors
        valid_results = [r for r in self.analysis_results if 'error' not in r]
        
        if not valid_results:
            self.logger.warning("No valid analysis results found")
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(valid_results)
        
        # Overall statistics
        total_contracts = len(valid_results)
        breached_contracts = len(df[df['breached'] == True])
        non_breached_contracts = total_contracts - breached_contracts
        
        # Return statistics
        avg_return = df['return_pct'].mean()
        median_return = df['return_pct'].median()
        positive_returns = len(df[df['return_pct'] > 0])
        
        # Breach statistics
        if breached_contracts > 0:
            breached_df = df[df['breached'] == True]
            avg_days_to_breach = breached_df['days_to_breach'].mean()
            median_days_to_breach = breached_df['days_to_breach'].median()
            avg_return_breached = breached_df['return_pct'].mean()
            min_days_to_breach = breached_df['days_to_breach'].min()
            max_days_to_breach = breached_df['days_to_breach'].max()
        else:
            avg_days_to_breach = None
            median_days_to_breach = None
            avg_return_breached = None
            min_days_to_breach = None
            max_days_to_breach = None
        
        if non_breached_contracts > 0:
            non_breached_df = df[df['breached'] == False]
            avg_return_non_breached = non_breached_df['return_pct'].mean()
        else:
            avg_return_non_breached = None
        
        # Strike distance analysis
        df['strike_distance_pct'] = ((df['strike'] - df['initial_price']) / df['initial_price'] * 100)
        
        summary = {
            'total_contracts': total_contracts,
            'breached_contracts': breached_contracts,
            'non_breached_contracts': non_breached_contracts,
            'breach_rate': (breached_contracts / total_contracts * 100) if total_contracts > 0 else 0,
            'avg_return_pct': avg_return,
            'median_return_pct': median_return,
            'positive_return_count': positive_returns,
            'positive_return_rate': (positive_returns / total_contracts * 100) if total_contracts > 0 else 0,
            'avg_days_to_breach': avg_days_to_breach,
            'median_days_to_breach': median_days_to_breach,
            'min_days_to_breach': min_days_to_breach,
            'max_days_to_breach': max_days_to_breach,
            'avg_return_breached': avg_return_breached,
            'avg_return_non_breached': avg_return_non_breached,
            'avg_strike_distance_pct': df['strike_distance_pct'].mean()
        }
        
        return summary
    
    def save_results(self):
        """Save analysis results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        if self.analysis_results:
            df = pd.DataFrame(self.analysis_results)
            detailed_path = os.path.join(self.output_dir, f'straddle_analysis_detailed_{timestamp}.csv')
            df.to_csv(detailed_path, index=False)
            self.logger.info(f"Detailed analysis saved to: {detailed_path}")
        
        # Save summary statistics
        summary = self.create_summary_statistics()
        if summary:
            summary_path = os.path.join(self.output_dir, f'straddle_summary_{timestamp}.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Summary statistics saved to: {summary_path}")
        
        return summary
    
    def print_summary_report(self, summary: dict):
        """Print comprehensive summary report."""
        print(f"\n{'='*80}")
        print("SHORT STRADDLE ANALYSIS SUMMARY (CORRECTED)")
        print(f"{'='*80}")
        
        print(f"Total contracts analyzed: {summary['total_contracts']:,}")
        print(f"Contracts where strike was reached: {summary['breached_contracts']:,} ({summary['breach_rate']:.1f}%)")
        print(f"Contracts where strike was NOT reached: {summary['non_breached_contracts']:,}")
        print(f"Average strike distance from initial spot: {summary['avg_strike_distance_pct']:.1f}%")
        print()
        
        print("RETURN ANALYSIS:")
        print(f"Average return: {summary['avg_return_pct']:.2f}%")
        print(f"Median return: {summary['median_return_pct']:.2f}%")
        print(f"Profitable trades: {summary['positive_return_count']}/{summary['total_contracts']} ({summary['positive_return_rate']:.1f}%)")
        print()
        
        if summary['avg_return_breached'] is not None:
            print("BREACH ANALYSIS:")
            print(f"Average days to breach: {summary['avg_days_to_breach']:.1f} days")
            print(f"Median days to breach: {summary['median_days_to_breach']:.1f} days")
            print(f"Fastest breach: {summary['min_days_to_breach']} days")
            print(f"Slowest breach: {summary['max_days_to_breach']} days")
            print(f"Average return when breached: {summary['avg_return_breached']:.2f}%")
        
        if summary['avg_return_non_breached'] is not None:
            print(f"Average return when NOT breached: {summary['avg_return_non_breached']:.2f}%")
        
        print()
        print("INTERPRETATION:")
        if summary['avg_return_pct'] > 0:
            print("✓ Short straddle strategy shows positive average returns")
        else:
            print("⚠ Short straddle strategy shows negative average returns")
        
        if summary['breach_rate'] < 30:
            print(f"✓ Low breach rate ({summary['breach_rate']:.1f}%) - strikes rarely reached")
        elif summary['breach_rate'] < 60:
            print(f"~ Moderate breach rate ({summary['breach_rate']:.1f}%) - mixed results")
        else:
            print(f"⚠ High breach rate ({summary['breach_rate']:.1f}%) - frequent large moves")
        
        print(f"\n{'='*80}")
    
    def run_analysis(self):
        """Run complete straddle analysis."""
        print("="*80)
        print("SHORT STRADDLE PERFORMANCE ANALYZER (CORRECTED)")
        print("="*80)
        print("Analyzing short straddle performance using actual spot price data:")
        print("- Checking if spot price actually reached strike levels")
        print("- Calculating returns for short straddle positions")
        print("- Using external spot prices for accurate breach detection")
        print(f"Straddle data: {self.straddle_data_dir}")
        print(f"Spot prices: {self.spot_prices_path}")
        print(f"Output: {self.output_dir}")
        print()
        
        # Run analysis
        self.analyze_all_straddles()
        
        # Save results and get summary
        summary = self.save_results()
        
        # Print report
        if summary:
            self.print_summary_report(summary)
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")

def main():
    # Configuration
    straddle_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    spot_prices_path = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/spot_prices.csv"
    output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_analysis_corrected"
    
    # Create analyzer and run analysis
    analyzer = ShortStraddleAnalyzer(
        straddle_data_dir=straddle_data_dir,
        spot_prices_path=spot_prices_path,
        output_dir=output_dir
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()