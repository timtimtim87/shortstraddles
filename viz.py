import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import glob
from typing import Tuple, Optional

class OptionsPlotter:
    def __init__(self, enhanced_data_dir: str):
        """
        Initialize the options plotter.
        
        Args:
            enhanced_data_dir: Directory containing enhanced CSV files
        """
        self.enhanced_data_dir = enhanced_data_dir
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create plots directory
        self.plots_dir = os.path.join(enhanced_data_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def load_contract_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare contract data."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def find_matching_put(self, call_filepath: str) -> Optional[str]:
        """Find the matching PUT contract for a given CALL contract."""
        # Parse the call filename to get matching put
        basename = os.path.basename(call_filepath)
        if '_CALL_enhanced.csv' not in basename:
            return None
        
        put_filename = basename.replace('_CALL_enhanced.csv', '_PUT_enhanced.csv')
        put_filepath = os.path.join(self.enhanced_data_dir, put_filename)
        
        if os.path.exists(put_filepath):
            return put_filepath
        return None
    
    def calculate_straddle_metrics(self, call_df: pd.DataFrame, put_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate straddle price and IV from call and put data."""
        # Merge on date
        straddle_df = pd.merge(call_df[['date', 'close', 'iv', 'volume', 'spot_price', 'time_to_expiry']], 
                              put_df[['date', 'close', 'iv', 'volume']], 
                              on='date', 
                              suffixes=('_call', '_put'))
        
        # Calculate straddle metrics
        straddle_df['straddle_price'] = straddle_df['close_call'] + straddle_df['close_put']
        
        # Calculate straddle IV as price-weighted average
        straddle_df['total_volume'] = straddle_df['volume_call'] + straddle_df['volume_put']
        
        # Handle cases where volume is 0
        mask = straddle_df['total_volume'] > 0
        straddle_df.loc[mask, 'straddle_iv'] = (
            (straddle_df.loc[mask, 'close_call'] * straddle_df.loc[mask, 'iv_call'] + 
             straddle_df.loc[mask, 'close_put'] * straddle_df.loc[mask, 'iv_put']) / 
            straddle_df.loc[mask, 'straddle_price']
        )
        
        # For zero volume, use simple average
        mask_zero = straddle_df['total_volume'] == 0
        straddle_df.loc[mask_zero, 'straddle_iv'] = (
            straddle_df.loc[mask_zero, 'iv_call'] + straddle_df.loc[mask_zero, 'iv_put']
        ) / 2
        
        # Calculate straddle OPP
        straddle_df['straddle_opp'] = (straddle_df['straddle_price'] / straddle_df['spot_price']) * 100
        
        return straddle_df
    
    def create_iv_plot(self, call_df: pd.DataFrame, contract_info: dict, 
                       put_df: pd.DataFrame = None, straddle_df: pd.DataFrame = None):
        """Create individual IV over time plot."""
        plt.figure(figsize=(12, 8))
        
        # Plot call IV
        valid_iv = call_df.dropna(subset=['iv'])
        if not valid_iv.empty:
            plt.plot(valid_iv['date'], valid_iv['iv'], 'b-', linewidth=3, 
                    label='Call IV', alpha=0.8, marker='o', markersize=4)
        
        # Plot put IV if available
        if put_df is not None:
            put_valid_iv = put_df.dropna(subset=['iv'])
            if not put_valid_iv.empty:
                plt.plot(put_valid_iv['date'], put_valid_iv['iv'], 'r-', linewidth=3, 
                        label='Put IV', alpha=0.8, marker='s', markersize=4)
        
        # Plot straddle IV if available
        if straddle_df is not None:
            straddle_valid_iv = straddle_df.dropna(subset=['straddle_iv'])
            if not straddle_valid_iv.empty:
                plt.plot(straddle_valid_iv['date'], straddle_valid_iv['straddle_iv'], 'g-', 
                        linewidth=3, label='Straddle IV', alpha=0.8, marker='^', markersize=4)
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Implied Volatility Over Time', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Implied Volatility (annualized)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_IV.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_price_plot(self, call_df: pd.DataFrame, contract_info: dict, 
                         put_df: pd.DataFrame = None, straddle_df: pd.DataFrame = None):
        """Create individual price over time plot."""
        plt.figure(figsize=(12, 8))
        
        # Plot call price
        plt.plot(call_df['date'], call_df['close'], 'b-', linewidth=3, 
                label='Call Price', alpha=0.8, marker='o', markersize=4)
        
        # Plot put price if available
        if put_df is not None:
            plt.plot(put_df['date'], put_df['close'], 'r-', linewidth=3, 
                    label='Put Price', alpha=0.8, marker='s', markersize=4)
        
        # Plot straddle price if available
        if straddle_df is not None:
            plt.plot(straddle_df['date'], straddle_df['straddle_price'], 'g-', 
                    linewidth=3, label='Straddle Price', alpha=0.8, marker='^', markersize=4)
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Option Prices Over Time', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_PRICE.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_opp_plot(self, call_df: pd.DataFrame, contract_info: dict, 
                       put_df: pd.DataFrame = None, straddle_df: pd.DataFrame = None):
        """Create individual OPP over time plot."""
        plt.figure(figsize=(12, 8))
        
        # Plot call OPP
        valid_opp = call_df.dropna(subset=['opp'])
        if not valid_opp.empty:
            plt.plot(valid_opp['date'], valid_opp['opp'], 'b-', linewidth=3, 
                    label='Call OPP', alpha=0.8, marker='o', markersize=4)
        
        # Plot put OPP if available
        if put_df is not None:
            put_valid_opp = put_df.dropna(subset=['opp'])
            if not put_valid_opp.empty:
                plt.plot(put_valid_opp['date'], put_valid_opp['opp'], 'r-', linewidth=3, 
                        label='Put OPP', alpha=0.8, marker='s', markersize=4)
        
        # Plot straddle OPP if available
        if straddle_df is not None:
            straddle_valid_opp = straddle_df.dropna(subset=['straddle_opp'])
            if not straddle_valid_opp.empty:
                plt.plot(straddle_valid_opp['date'], straddle_valid_opp['straddle_opp'], 'g-', 
                        linewidth=3, label='Straddle OPP', alpha=0.8, marker='^', markersize=4)
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Options Price Percentage (OPP)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('OPP (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_OPP.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_volume_plot(self, call_df: pd.DataFrame, contract_info: dict, put_df: pd.DataFrame = None):
        """Create individual volume plot."""
        plt.figure(figsize=(12, 8))
        
        # Plot call volume
        plt.bar(call_df['date'], call_df['volume'], alpha=0.7, color='blue', 
               label='Call Volume', width=1)
        
        # Plot put volume if available
        if put_df is not None:
            plt.bar(put_df['date'], put_df['volume'], alpha=0.7, color='red', 
                   label='Put Volume', width=1)
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Trading Volume', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Volume', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_VOLUME.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_spot_moneyness_plot(self, call_df: pd.DataFrame, contract_info: dict):
        """Create individual spot price and moneyness plot."""
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Spot price on left axis
        ax1.plot(call_df['date'], call_df['spot_price'], 'black', linewidth=3, 
                label='Spot Price', marker='o', markersize=4)
        ax1.axhline(y=contract_info['strike'], color='gray', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f"Strike ${contract_info['strike']}")
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Moneyness on right axis
        ax2 = ax1.twinx()
        valid_moneyness = call_df.dropna(subset=['moneyness'])
        if not valid_moneyness.empty:
            ax2.plot(valid_moneyness['date'], valid_moneyness['moneyness'], 'orange', 
                    linewidth=3, label='Moneyness', alpha=0.8, marker='^', markersize=4)
            ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, linewidth=2)
            ax2.set_ylabel('Moneyness (Spot/Strike)', fontsize=12, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Spot Price and Moneyness', 
                 fontsize=16, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_SPOT_MONEYNESS.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_time_decay_plot(self, call_df: pd.DataFrame, contract_info: dict):
        """Create individual time to expiry plot."""
        plt.figure(figsize=(12, 8))
        
        plt.plot(call_df['date'], call_df['time_to_expiry'] * 365, 'purple', 
                linewidth=3, alpha=0.8, marker='o', markersize=4)
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Time to Expiry', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Days to Expiry', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_TIME_DECAY.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_straddle_price_plot(self, straddle_df: pd.DataFrame, contract_info: dict):
        """Create individual straddle price plot."""
        if straddle_df is None or len(straddle_df) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(straddle_df['date'], straddle_df['straddle_price'], 'green', 
                linewidth=3, alpha=0.8, marker='o', markersize=4, label='Straddle Price')
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Straddle Price Over Time', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Straddle Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_STRADDLE_PRICE.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_straddle_iv_plot(self, straddle_df: pd.DataFrame, contract_info: dict):
        """Create individual straddle IV plot."""
        if straddle_df is None or len(straddle_df) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        straddle_valid_iv = straddle_df.dropna(subset=['straddle_iv'])
        if not straddle_valid_iv.empty:
            plt.plot(straddle_valid_iv['date'], straddle_valid_iv['straddle_iv'], 'green', 
                    linewidth=3, alpha=0.8, marker='o', markersize=4, label='Straddle IV')
        
        plt.title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry"]} - Straddle IV Over Time', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Straddle Implied Volatility', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{contract_info['symbol']}_{contract_info['expiry']}_{int(contract_info['strike'])}_STRADDLE_IV.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def analyze_contract(self, call_filepath: str):
        """Analyze a single call contract and create all individual plots."""
        print(f"\nAnalyzing: {os.path.basename(call_filepath)}")
        
        # Load call data
        call_df = self.load_contract_data(call_filepath)
        if call_df is None:
            return
        
        # Extract contract info
        basename = os.path.basename(call_filepath)
        parts = basename.replace('_enhanced.csv', '').split('_')
        contract_info = {
            'symbol': parts[0],
            'expiry': parts[1],
            'strike': float(parts[2]),
            'type': parts[3]
        }
        
        print(f"Contract: {contract_info['symbol']} {contract_info['expiry']} ${contract_info['strike']} {contract_info['type']}")
        print(f"Data range: {call_df['date'].min().strftime('%Y-%m-%d')} to {call_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total observations: {len(call_df)}")
        
        # Find and load matching put
        put_filepath = self.find_matching_put(call_filepath)
        put_df = None
        straddle_df = None
        
        if put_filepath:
            put_df = self.load_contract_data(put_filepath)
            if put_df is not None:
                print(f"Found matching PUT: {os.path.basename(put_filepath)}")
                straddle_df = self.calculate_straddle_metrics(call_df, put_df)
                print(f"Straddle data points: {len(straddle_df)}")
            else:
                print("Could not load matching PUT data")
        else:
            print("No matching PUT contract found")
        
        print(f"\nCreating individual plots in: {self.plots_dir}")
        
        # Create all individual plots
        self.create_iv_plot(call_df, contract_info, put_df, straddle_df)
        self.create_price_plot(call_df, contract_info, put_df, straddle_df)
        self.create_opp_plot(call_df, contract_info, put_df, straddle_df)
        self.create_volume_plot(call_df, contract_info, put_df)
        self.create_spot_moneyness_plot(call_df, contract_info)
        self.create_time_decay_plot(call_df, contract_info)
        
        # Create straddle-specific plots if straddle data exists
        if straddle_df is not None:
            self.create_straddle_price_plot(straddle_df, contract_info)
            self.create_straddle_iv_plot(straddle_df, contract_info)
        
        # Print summary statistics
        if straddle_df is not None and len(straddle_df) > 0:
            print("\nStraddle Summary Statistics:")
            print(f"  Average straddle price: ${straddle_df['straddle_price'].mean():.2f}")
            print(f"  Straddle price range: ${straddle_df['straddle_price'].min():.2f} - ${straddle_df['straddle_price'].max():.2f}")
            print(f"  Average straddle IV: {straddle_df['straddle_iv'].mean():.3f}")
            print(f"  Average straddle OPP: {straddle_df['straddle_opp'].mean():.2f}%")
        
        print(f"\nAll plots saved successfully to: {self.plots_dir}")
        
        return {
            'call_df': call_df,
            'put_df': put_df,
            'straddle_df': straddle_df,
            'contract_info': contract_info
        }

def main():
    # Configuration
    enhanced_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/options_data_enhanced"
    
    # Create plotter
    plotter = OptionsPlotter(enhanced_data_dir)
    
    # Analyze specific contract
    target_file = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/options_data_enhanced/AAPL_2025-01-17_210_CALL_enhanced.csv"
    
    if os.path.exists(target_file):
        result = plotter.analyze_contract(target_file)
    else:
        print(f"File not found: {target_file}")
        
        # Show available files
        pattern = os.path.join(enhanced_data_dir, "*_CALL_enhanced.csv")
        available_files = glob.glob(pattern)
        print(f"\nAvailable CALL files in {enhanced_data_dir}:")
        for file in sorted(available_files):
            print(f"  {os.path.basename(file)}")
    
    # Optional: Analyze all contracts
    analyze_all = input("\nWould you like to analyze all contracts? (y/n): ").lower()
    
    if analyze_all == 'y':
        pattern = os.path.join(enhanced_data_dir, "*_CALL_enhanced.csv")
        call_files = glob.glob(pattern)
        
        for call_file in sorted(call_files):
            print(f"\n{'='*60}")
            result = plotter.analyze_contract(call_file)

if __name__ == "__main__":
    main()