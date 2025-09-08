import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Tuple, Optional
import logging
from scipy.stats import norm
import math

class StraddleDataCollector:
    def __init__(self, config_path: str = "config.json", spot_prices_path: Optional[str] = None):
        """Initialize the straddle data collector with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']
        self.symbols = self.config['symbols']
        self.expiry_dates = self.config['expiry_dates']
        self.otm_percentages = self.config['otm_percentages']
        self.rate_limit_delay = self.config['rate_limit_delay']
        self.output_dir = self.config['output_directory']
        self.calendar_year_offset = self.config.get('calendar_year_offset', 365)
        
        # Load spot prices if provided
        self.spot_prices_df = None
        if spot_prices_path and os.path.exists(spot_prices_path):
            try:
                self.spot_prices_df = pd.read_csv(spot_prices_path)
                self.spot_prices_df['Date'] = pd.to_datetime(self.spot_prices_df['Date'])
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"Loaded spot prices from {spot_prices_path}")
            except Exception as e:
                self.logger = logging.getLogger(__name__)
                self.logger.warning(f"Could not load spot prices from {spot_prices_path}: {e}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_spot_price_from_csv(self, symbol: str, date: str) -> Optional[float]:
        """Get spot price from loaded CSV file."""
        if self.spot_prices_df is None:
            return None
        
        try:
            date_dt = pd.to_datetime(date)
            filtered_df = self.spot_prices_df[
                (self.spot_prices_df['Ticker'] == symbol) & 
                (self.spot_prices_df['Date'] == date_dt)
            ]
            
            if not filtered_df.empty:
                return float(filtered_df.iloc[0]['Close'])
            else:
                # Try to find the closest date within a reasonable range
                symbol_data = self.spot_prices_df[self.spot_prices_df['Ticker'] == symbol]
                if not symbol_data.empty:
                    symbol_data['date_diff'] = abs(symbol_data['Date'] - date_dt)
                    closest = symbol_data.loc[symbol_data['date_diff'].idxmin()]
                    if closest['date_diff'].days <= 7:  # Within a week
                        self.logger.info(f"Using closest date {closest['Date'].strftime('%Y-%m-%d')} for {symbol} (requested: {date})")
                        return float(closest['Close'])
                
                return None
        except Exception as e:
            self.logger.error(f"Error reading spot price from CSV for {symbol} on {date}: {e}")
            return None

    def get_spot_price_historical(self, symbol: str, date: str) -> Optional[float]:
        """Get historical spot price for a given symbol and date."""
        # First try to get from CSV if available
        csv_price = self.get_spot_price_from_csv(symbol, date)
        if csv_price is not None:
            return csv_price
        
        # Fall back to API
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/day/{date}/{date}"
        params = {"apikey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('resultsCount', 0) > 0:
                # Use closing price as spot price
                return float(data['results'][0]['c'])
            else:
                self.logger.warning(f"No data found for {symbol} on {date}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching spot price for {symbol} on {date}: {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)

    def get_available_contracts(self, symbol: str, expiry_date: str, contract_type: str = None) -> List[Dict]:
        """Get all available options contracts for a given symbol and expiry date."""
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": symbol,
            "expiration_date": expiry_date,
            "limit": 1000,
            "apikey": self.api_key
        }
        
        if contract_type:
            params["contract_type"] = contract_type
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            contracts = data.get('results', [])
            self.logger.info(f"Found {len(contracts)} {contract_type or 'total'} contracts for {symbol} expiring {expiry_date}")
            
            return contracts
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching contracts for {symbol} {expiry_date}: {e}")
            return []
        
        finally:
            time.sleep(self.rate_limit_delay)

    def filter_contracts_by_increment(self, contracts: List[Dict], increment: float = 10.0) -> List[Dict]:
        """Filter contracts to only include strikes at specified increments."""
        filtered = []
        
        for contract in contracts:
            strike = contract.get('strike_price', 0)
            # Check if strike is a multiple of the increment
            if strike % increment == 0:
                filtered.append(contract)
        
        self.logger.info(f"Filtered to {len(filtered)} contracts with ${increment} strike increments")
        return filtered

    def find_target_strikes(self, spot_price: float, otm_percentages: List[float], increment: float = 10.0) -> List[float]:
        """Find target strikes based on spot price and OTM percentages, rounded to increments."""
        target_strikes = []
        
        # ATM (0% OTM)
        atm_strike = round(spot_price / increment) * increment
        target_strikes.append(atm_strike)
        
        # OTM strikes
        for otm_pct in otm_percentages:
            if otm_pct > 0:  # Only positive OTM percentages
                otm_price = spot_price * (1 + otm_pct)
                otm_strike = round(otm_price / increment) * increment
                target_strikes.append(otm_strike)
        
        return sorted(set(target_strikes))  # Remove duplicates and sort

    def get_contract_daily_data(self, contract_ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get daily OHLCV data for a specific options contract."""
        # Remove the 'O:' prefix if present
        if contract_ticker.startswith('O:'):
            contract_ticker = contract_ticker[2:]
        
        url = f"{self.base_url}/v2/aggs/ticker/O:{contract_ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            "apikey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('resultsCount', 0) > 0:
                results = []
                for result in data['results']:
                    timestamp = result.get('t')
                    if timestamp:
                        date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                        results.append({
                            'ticker': f"O:{contract_ticker}",
                            'date': date,
                            'timestamp': timestamp,
                            'open': result.get('o'),
                            'high': result.get('h'),
                            'low': result.get('l'),
                            'close': result.get('c'),
                            'volume': result.get('v', 0),
                            'vwap': result.get('vw'),
                            'transactions': result.get('n', 0)
                        })
                
                if results:
                    df = pd.DataFrame(results)
                    self.logger.info(f"Retrieved {len(df)} days of data for {contract_ticker}")
                    return df
                else:
                    self.logger.warning(f"No valid results for {contract_ticker}")
                    return None
            else:
                self.logger.warning(f"No data found for {contract_ticker} between {start_date} and {end_date}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching daily data for {contract_ticker}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing daily data for {contract_ticker}: {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)

    def generate_historical_contract_tickers(self, symbol: str, expiry_date: str, strikes: List[float]) -> Dict[str, str]:
        """Generate contract tickers for expired options based on known format."""
        contract_tickers = {}
        
        # Parse expiry date to create option symbol format
        expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        exp_str = expiry_dt.strftime('%y%m%d')  # YYMMDD format
        
        for strike in strikes:
            # Format strike as 8-digit integer (multiply by 1000 for 3 decimal places)
            strike_str = f"{int(strike * 1000):08d}"
            
            # Generate call and put tickers
            call_ticker = f"{symbol}{exp_str}C{strike_str}"
            put_ticker = f"{symbol}{exp_str}P{strike_str}"
            
            contract_tickers[f"{strike}_CALL"] = call_ticker
            contract_tickers[f"{strike}_PUT"] = put_ticker
        
        return contract_tickers

    def collect_contract_time_series(self, symbol: str, expiry_date: str) -> Dict[str, pd.DataFrame]:
        """Collect time series data for contracts at target strikes."""
        self.logger.info(f"Collecting contract time series for {symbol} expiring {expiry_date}")
        
        contract_data = {}
        
        try:
            # Check if this is an expired contract
            expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
            is_expired = expiry_dt < datetime.now()
            
            if is_expired:
                self.logger.info(f"Collecting historical data for expired expiry: {expiry_date}")
                
                # For expired contracts, use a different strategy
                # Calculate reference date for determining target strikes (1 year before expiry or use earliest available data)
                ref_dt = expiry_dt - timedelta(days=self.calendar_year_offset)
                ref_date = ref_dt.strftime('%Y-%m-%d')
                
                # Get spot price for reference date
                spot_price = self.get_spot_price_historical(symbol, ref_date)
                if not spot_price:
                    # If we can't get spot price from 1 year ago, try closer dates
                    for days_back in [300, 200, 100, 50]:
                        alt_ref_dt = expiry_dt - timedelta(days=days_back)
                        alt_ref_date = alt_ref_dt.strftime('%Y-%m-%d')
                        spot_price = self.get_spot_price_historical(symbol, alt_ref_date)
                        if spot_price:
                            ref_date = alt_ref_date
                            self.logger.info(f"Using alternative reference date: {ref_date}")
                            break
                
                if not spot_price:
                    self.logger.error(f"Could not get any spot price for {symbol} around {expiry_date}")
                    return contract_data
                
                self.logger.info(f"Reference spot price on {ref_date}: ${spot_price:.2f}")
                
                # Find target strikes based on spot price and OTM percentages
                target_strikes = self.find_target_strikes(spot_price, self.otm_percentages, increment=10.0)
                self.logger.info(f"Target strikes: {target_strikes}")
                
                # Generate contract tickers for expired options
                contract_tickers = self.generate_historical_contract_tickers(symbol, expiry_date, target_strikes)
                
                # Try to collect data for each generated ticker
                for strike in target_strikes:
                    # Try calls
                    call_ticker = contract_tickers.get(f"{strike}_CALL")
                    if call_ticker:
                        self.logger.info(f"Trying to collect call data for ${strike} strike: {call_ticker}")
                        
                        call_data = self.get_contract_daily_data(
                            call_ticker, ref_date, expiry_date
                        )
                        
                        if call_data is not None and not call_data.empty:
                            contract_key = f"{symbol}_{expiry_date}_{int(strike)}_CALL"
                            contract_data[contract_key] = call_data
                            self.logger.info(f"Collected {len(call_data)} days for call ${strike}")
                        else:
                            self.logger.warning(f"No data found for call ticker: {call_ticker}")
                    
                    # Try puts
                    put_ticker = contract_tickers.get(f"{strike}_PUT")
                    if put_ticker:
                        self.logger.info(f"Trying to collect put data for ${strike} strike: {put_ticker}")
                        
                        put_data = self.get_contract_daily_data(
                            put_ticker, ref_date, expiry_date
                        )
                        
                        if put_data is not None and not put_data.empty:
                            contract_key = f"{symbol}_{expiry_date}_{int(strike)}_PUT"
                            contract_data[contract_key] = put_data
                            self.logger.info(f"Collected {len(put_data)} days for put ${strike}")
                        else:
                            self.logger.warning(f"No data found for put ticker: {put_ticker}")
                    
                    # Add delay between strikes
                    time.sleep(self.rate_limit_delay)
                
            else:
                # For active contracts, use the original method
                # Calculate reference date for determining target strikes (1 year before expiry)
                ref_dt = expiry_dt - timedelta(days=self.calendar_year_offset)
                ref_date = ref_dt.strftime('%Y-%m-%d')
                
                # Get spot price for reference date
                spot_price = self.get_spot_price_historical(symbol, ref_date)
                if not spot_price:
                    self.logger.error(f"Could not get spot price for {symbol} on {ref_date}")
                    return contract_data
                
                self.logger.info(f"Reference spot price on {ref_date}: ${spot_price:.2f}")
                
                # Find target strikes based on spot price and OTM percentages
                target_strikes = self.find_target_strikes(spot_price, self.otm_percentages, increment=10.0)
                self.logger.info(f"Target strikes: {target_strikes}")
                
                # Get all available contracts
                call_contracts = self.get_available_contracts(symbol, expiry_date, "call")
                put_contracts = self.get_available_contracts(symbol, expiry_date, "put")
                
                # Filter by increment
                filtered_calls = self.filter_contracts_by_increment(call_contracts, 10.0)
                filtered_puts = self.filter_contracts_by_increment(put_contracts, 10.0)
                
                # Create strike-to-contract mapping
                call_by_strike = {contract['strike_price']: contract for contract in filtered_calls}
                put_by_strike = {contract['strike_price']: contract for contract in filtered_puts}
                
                # Collect data for each target strike
                for strike in target_strikes:
                    if strike in call_by_strike:
                        call_contract = call_by_strike[strike]
                        self.logger.info(f"Collecting call data for ${strike} strike")
                        
                        call_data = self.get_contract_daily_data(
                            call_contract['ticker'], ref_date, expiry_date
                        )
                        
                        if call_data is not None and not call_data.empty:
                            contract_key = f"{symbol}_{expiry_date}_{int(strike)}_CALL"
                            contract_data[contract_key] = call_data
                            self.logger.info(f"Collected {len(call_data)} days for call ${strike}")
                    
                    if strike in put_by_strike:
                        put_contract = put_by_strike[strike]
                        self.logger.info(f"Collecting put data for ${strike} strike")
                        
                        put_data = self.get_contract_daily_data(
                            put_contract['ticker'], ref_date, expiry_date
                        )
                        
                        if put_data is not None and not put_data.empty:
                            contract_key = f"{symbol}_{expiry_date}_{int(strike)}_PUT"
                            contract_data[contract_key] = put_data
                            self.logger.info(f"Collected {len(put_data)} days for put ${strike}")
                    
                    # Add delay between strikes
                    time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            self.logger.error(f"Error collecting contract time series for {symbol} {expiry_date}: {e}")
        
        return contract_data

    def save_contract_files(self, all_contract_data: Dict[str, pd.DataFrame]) -> None:
        """Save each contract's time series data to a separate CSV file."""
        for contract_key, df in all_contract_data.items():
            filename = f"{contract_key}_timeseries.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                df.to_csv(filepath, index=False)
                self.logger.info(f"Saved {filename} with {len(df)} rows")
            except Exception as e:
                self.logger.error(f"Error saving {filename}: {e}")

    def run_collection(self) -> None:
        """Run the complete data collection process for time series data."""
        self.logger.info("Starting options time series data collection")
        
        all_contract_data = {}
        
        for symbol in self.symbols:
            for expiry_date in self.expiry_dates:
                try:
                    contract_data = self.collect_contract_time_series(symbol, expiry_date)
                    if contract_data:
                        all_contract_data.update(contract_data)
                        self.logger.info(f"Collected {len(contract_data)} contract time series for {symbol} {expiry_date}")
                    else:
                        self.logger.warning(f"No contract data collected for {symbol} {expiry_date}")
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {expiry_date}: {e}")
                
                # Add delay between expiry dates
                time.sleep(self.rate_limit_delay * 10)
        
        if all_contract_data:
            # Save each contract to its own file
            self.save_contract_files(all_contract_data)
            
            self.logger.info(f"Time series collection complete. Saved {len(all_contract_data)} contract files")
            
            # Display summary
            print("\n" + "="*80)
            print("COLLECTION SUMMARY")
            print("="*80)
            print(f"Total contracts collected: {len(all_contract_data)}")
            
            for contract_key, df in all_contract_data.items():
                parts = contract_key.split('_')
                symbol = parts[0]
                expiry = parts[1]
                strike = parts[2]
                option_type = parts[3]
                print(f"  {symbol} {expiry} ${strike} {option_type}: {len(df)} trading days")
            
        else:
            self.logger.warning("No contract time series data was collected")

if __name__ == "__main__":
    # You can specify a custom path to your config file and spot prices file here
    config_path = "config.json"
    spot_prices_path = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/spot_prices.csv"
    
    collector = StraddleDataCollector(config_path=config_path, spot_prices_path=spot_prices_path)
    collector.run_collection()