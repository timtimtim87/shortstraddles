import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import glob
from datetime import datetime
import logging

class OptionsAnalyzer:
    def __init__(self, options_data_dir: str, spot_prices_path: str, output_dir: str = None):
        """
        Initialize the options analyzer.
        
        Args:
            options_data_dir: Directory containing the options time series CSV files
            spot_prices_path: Path to the spot prices CSV file
            output_dir: Directory to save enhanced files (defaults to options_data_dir + '_enhanced')
        """
        self.options_data_dir = options_data_dir
        self.spot_prices_path = spot_prices_path
        self.output_dir = output_dir or f"{options_data_dir}_enhanced"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load spot prices
        self.load_spot_prices()
    
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
    
    def black_scholes_iv(self, option_price: float, spot_price: float, strike: float, 
                        time_to_expiry: float, option_type: str, risk_free_rate: float = 0.02) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            spot_price: Current price of underlying asset
            strike: Strike price of the option
            time_to_expiry: Time to expiry in years
            option_type: 'C' for call, 'P' for put
            risk_free_rate: Risk-free interest rate (default 2%)
        
        Returns:
            Implied volatility or None if calculation fails
        """
        if option_price <= 0 or spot_price <= 0 or strike <= 0 or time_to_expiry <= 0:
            return None
        
        # Check for extreme conditions
        if time_to_expiry < 1/365:  # Less than 1 day
            return None
        
        # Check for extreme moneyness
        moneyness = spot_price / strike if option_type.upper() == 'C' else strike / spot_price
        if moneyness < 0.3 or moneyness > 3.0:
            return None
        
        # Initial guess for volatility
        vol = 0.3
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            try:
                d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
                d2 = d1 - vol * np.sqrt(time_to_expiry)
                
                if option_type.upper() == 'C':
                    option_price_calc = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
                else:
                    option_price_calc = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
                
                vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
                
                price_diff = option_price_calc - option_price
                
                if abs(price_diff) < tolerance:
                    return vol if 0.01 <= vol <= 5.0 else None
                
                if vega == 0 or vega < 1e-10:
                    break
                    
                vol = vol - price_diff / vega
                
                # Keep volatility within reasonable bounds
                vol = max(0.01, min(vol, 5.0))
                
            except (ValueError, FloatingPointError, OverflowError, np.core._exceptions._ArrayMemoryError):
                return None
        
        return vol if 0.01 <= vol <= 5.0 else None
    
    def calculate_time_to_expiry(self, current_date: str, expiry_date: str) -> float:
        """Calculate time to expiry in years."""
        try:
            current_dt = pd.to_datetime(current_date)
            expiry_dt = pd.to_datetime(expiry_date)
            
            days_to_expiry = (expiry_dt - current_dt).days
            return max(days_to_expiry / 365.25, 1/365.25)  # Minimum 1 day
            
        except Exception as e:
            self.logger.error(f"Error calculating time to expiry: {e}")
            return None
    
    def parse_contract_info(self, filename: str) -> dict:
        """Extract contract information from filename."""
        # Example filename: AAPL_2025-01-17_190_CALL_timeseries.csv
        try:
            basename = os.path.basename(filename).replace('_timeseries.csv', '')
            parts = basename.split('_')
            
            return {
                'symbol': parts[0],
                'expiry_date': parts[1],
                'strike': float(parts[2]),
                'option_type': parts[3]
            }
        except Exception as e:
            self.logger.error(f"Error parsing filename {filename}: {e}")
            return None
    
    def process_options_file(self, filepath: str) -> pd.DataFrame:
        """Process a single options time series file to add IV and OPP."""
        self.logger.info(f"Processing {os.path.basename(filepath)}")
        
        # Parse contract info from filename
        contract_info = self.parse_contract_info(filepath)
        if not contract_info:
            self.logger.error(f"Could not parse contract info from {filepath}")
            return None
        
        # Load the options data
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None
        
        # Add contract info columns
        df['symbol'] = contract_info['symbol']
        df['expiry_date'] = contract_info['expiry_date']
        df['strike'] = contract_info['strike']
        df['option_type'] = contract_info['option_type']
        
        # Initialize new columns
        df['spot_price'] = np.nan
        df['time_to_expiry'] = np.nan
        df['iv'] = np.nan
        df['opp'] = np.nan
        df['moneyness'] = np.nan
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                date_str = row['date'].strftime('%Y-%m-%d')
                
                # Get spot price
                spot_price = self.get_spot_price(contract_info['symbol'], date_str)
                if spot_price is None:
                    continue
                
                df.at[idx, 'spot_price'] = spot_price
                
                # Calculate time to expiry
                time_to_expiry = self.calculate_time_to_expiry(date_str, contract_info['expiry_date'])
                if time_to_expiry is None:
                    continue
                
                df.at[idx, 'time_to_expiry'] = time_to_expiry
                
                # Calculate moneyness
                if contract_info['option_type'] == 'CALL':
                    moneyness = spot_price / contract_info['strike']
                else:  # PUT
                    moneyness = contract_info['strike'] / spot_price
                
                df.at[idx, 'moneyness'] = moneyness
                
                # Calculate OPP (Options Price Percentage)
                if row['close'] > 0 and spot_price > 0:
                    opp = (row['close'] / spot_price) * 100
                    df.at[idx, 'opp'] = opp
                
                # Calculate IV
                if row['close'] > 0 and spot_price > 0:
                    iv = self.black_scholes_iv(
                        option_price=row['close'],
                        spot_price=spot_price,
                        strike=contract_info['strike'],
                        time_to_expiry=time_to_expiry,
                        option_type=contract_info['option_type'][0]  # 'C' or 'P'
                    )
                    
                    if iv is not None:
                        df.at[idx, 'iv'] = iv
                
            except Exception as e:
                self.logger.warning(f"Error processing row {idx} in {filepath}: {e}")
                continue
        
        # Add some summary statistics
        valid_iv_count = df['iv'].notna().sum()
        total_rows = len(df)
        
        self.logger.info(f"  Processed {total_rows} rows, calculated IV for {valid_iv_count} ({valid_iv_count/total_rows*100:.1f}%)")
        
        return df
    
    def process_all_files(self):
        """Process all options time series files in the directory."""
        # Find all CSV files
        pattern = os.path.join(self.options_data_dir, "*_timeseries.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            self.logger.error(f"No time series CSV files found in {self.options_data_dir}")
            return
        
        self.logger.info(f"Found {len(csv_files)} files to process")
        
        successful_files = 0
        
        for filepath in csv_files:
            try:
                enhanced_df = self.process_options_file(filepath)
                
                if enhanced_df is not None:
                    # Save enhanced file
                    filename = os.path.basename(filepath).replace('_timeseries.csv', '_enhanced.csv')
                    output_path = os.path.join(self.output_dir, filename)
                    
                    enhanced_df.to_csv(output_path, index=False)
                    successful_files += 1
                    
                    self.logger.info(f"  Saved enhanced file: {filename}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filepath}: {e}")
        
        self.logger.info(f"Successfully processed {successful_files}/{len(csv_files)} files")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of all processed files."""
        try:
            # Load all enhanced files
            pattern = os.path.join(self.output_dir, "*_enhanced.csv")
            enhanced_files = glob.glob(pattern)
            
            if not enhanced_files:
                self.logger.warning("No enhanced files found for summary report")
                return
            
            summary_data = []
            
            for filepath in enhanced_files:
                try:
                    df = pd.read_csv(filepath)
                    
                    if len(df) == 0:
                        continue
                    
                    # Extract contract info
                    contract_info = {
                        'symbol': df['symbol'].iloc[0],
                        'expiry_date': df['expiry_date'].iloc[0],
                        'strike': df['strike'].iloc[0],
                        'option_type': df['option_type'].iloc[0]
                    }
                    
                    # Calculate statistics
                    stats = {
                        'file': os.path.basename(filepath),
                        'symbol': contract_info['symbol'],
                        'expiry_date': contract_info['expiry_date'],
                        'strike': contract_info['strike'],
                        'option_type': contract_info['option_type'],
                        'total_rows': len(df),
                        'valid_iv_count': df['iv'].notna().sum(),
                        'valid_opp_count': df['opp'].notna().sum(),
                        'avg_iv': df['iv'].mean(),
                        'median_iv': df['iv'].median(),
                        'avg_opp': df['opp'].mean(),
                        'median_opp': df['opp'].median(),
                        'avg_volume': df['volume'].mean(),
                        'total_volume': df['volume'].sum(),
                        'price_range': f"${df['close'].min():.2f} - ${df['close'].max():.2f}",
                        'date_range': f"{df['date'].min()} to {df['date'].max()}"
                    }
                    
                    summary_data.append(stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {filepath} for summary: {e}")
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(self.output_dir, 'summary_report.csv')
                summary_df.to_csv(summary_path, index=False)
                
                self.logger.info(f"Summary report saved to: {summary_path}")
                
                # Print summary to console
                print("\n" + "="*100)
                print("OPTIONS ANALYSIS SUMMARY")
                print("="*100)
                
                for _, row in summary_df.iterrows():
                    print(f"{row['symbol']} {row['expiry_date']} ${row['strike']:.0f} {row['option_type']}")
                    print(f"  Rows: {row['total_rows']}, IV: {row['valid_iv_count']} ({row['valid_iv_count']/row['total_rows']*100:.1f}%)")
                    print(f"  Avg IV: {row['avg_iv']:.3f}, Avg OPP: {row['avg_opp']:.3f}%")
                    print(f"  Volume: {row['total_volume']:,.0f}, Price Range: {row['price_range']}")
                    print()
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")

if __name__ == "__main__":
    # Configuration
    options_data_dir = "./options_data"
    spot_prices_path = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/spot_prices.csv"
    output_dir = "./options_data_enhanced"
    
    # Create analyzer and process files
    analyzer = OptionsAnalyzer(
        options_data_dir=options_data_dir,
        spot_prices_path=spot_prices_path,
        output_dir=output_dir
    )
    
    analyzer.process_all_files()