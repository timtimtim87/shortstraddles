import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime
import logging

class StraddlePlotter:
    def __init__(self, straddle_data_dir: str, plots_output_dir: str = None):
        """
        Initialize the straddle plotter.
        
        Args:
            straddle_data_dir: Directory containing straddle CSV files
            plots_output_dir: Directory to save plots (defaults to straddle_data_dir + '_plots')
        """
        self.straddle_data_dir = straddle_data_dir
        self.plots_output_dir = plots_output_dir or f"{straddle_data_dir}_plots"
        
        # Create plots directory
        os.makedirs(self.plots_output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Set up clean plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
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
    
    def create_straddle_plot(self, straddle_df: pd.DataFrame, contract_info: dict, filename: str):
        """Create a dual-panel plot showing straddle price and IV over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Filter out any invalid data
        valid_price_data = straddle_df.dropna(subset=['straddle_price'])
        valid_iv_data = straddle_df.dropna(subset=['straddle_iv'])
        
        # Plot 1: Straddle Price Over Time
        if not valid_price_data.empty:
            ax1.plot(valid_price_data['date'], valid_price_data['straddle_price'], 
                    'b-', linewidth=2.5, marker='o', markersize=3, alpha=0.8)
            ax1.set_title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry_date"]} - Straddle Price', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel('Straddle Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add some styling
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Show price range in title
            min_price = valid_price_data['straddle_price'].min()
            max_price = valid_price_data['straddle_price'].max()
            ax1.text(0.02, 0.98, f'Range: ${min_price:.2f} - ${max_price:.2f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No valid price data', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
            ax1.set_title(f'{contract_info["symbol"]} ${contract_info["strike"]} {contract_info["expiry_date"]} - Straddle Price', 
                         fontsize=14, fontweight='bold')
        
        # Plot 2: Straddle IV Over Time
        if not valid_iv_data.empty:
            ax2.plot(valid_iv_data['date'], valid_iv_data['straddle_iv'], 
                    'g-', linewidth=2.5, marker='s', markersize=3, alpha=0.8)
            ax2.set_title('Straddle Implied Volatility', fontsize=14, fontweight='bold', pad=20)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Implied Volatility', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add some styling
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Show IV range in title
            min_iv = valid_iv_data['straddle_iv'].min()
            max_iv = valid_iv_data['straddle_iv'].max()
            avg_iv = valid_iv_data['straddle_iv'].mean()
            ax2.text(0.02, 0.98, f'Range: {min_iv:.3f} - {max_iv:.3f} (Avg: {avg_iv:.3f})', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No valid IV data', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
            ax2.set_title('Straddle Implied Volatility', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
        
        # Overall plot formatting
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save plot
        plot_filename = filename.replace('_STRADDLE.csv', '_straddle_plot.png')
        plot_path = os.path.join(self.plots_output_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved plot: {plot_filename}")
        return plot_path
    
    def plot_all_straddles(self):
        """Create plots for all straddle files."""
        self.logger.info(f"Starting to create straddle plots...")
        self.logger.info(f"Looking for files in: {self.straddle_data_dir}")
        self.logger.info(f"Saving plots to: {self.plots_output_dir}")
        
        # Find all straddle files
        pattern = os.path.join(self.straddle_data_dir, "*_STRADDLE.csv")
        straddle_files = glob.glob(pattern)
        
        if not straddle_files:
            self.logger.error(f"No straddle files found in {self.straddle_data_dir}")
            return
        
        self.logger.info(f"Found {len(straddle_files)} straddle files")
        
        successful_plots = 0
        failed_plots = 0
        
        for filepath in straddle_files:
            filename = os.path.basename(filepath)
            
            try:
                # Load data
                straddle_df = self.load_straddle_data(filepath)
                if straddle_df is None or straddle_df.empty:
                    self.logger.warning(f"Skipping {filename}: No data")
                    failed_plots += 1
                    continue
                
                # Parse contract info
                contract_info = self.parse_straddle_info(filename)
                if contract_info is None:
                    self.logger.warning(f"Skipping {filename}: Could not parse contract info")
                    failed_plots += 1
                    continue
                
                # Create plot
                plot_path = self.create_straddle_plot(straddle_df, contract_info, filename)
                successful_plots += 1
                
            except Exception as e:
                self.logger.error(f"Error creating plot for {filename}: {e}")
                failed_plots += 1
        
        # Summary
        self.logger.info(f"Plot creation complete!")
        self.logger.info(f"Successful plots: {successful_plots}")
        self.logger.info(f"Failed plots: {failed_plots}")
        self.logger.info(f"All plots saved to: {self.plots_output_dir}")
        
        # Print summary by symbol
        self.create_summary_report(successful_plots, failed_plots)
    
    def create_summary_report(self, successful_plots: int, failed_plots: int):
        """Create a summary report of created plots."""
        try:
            # Count plots by symbol
            plot_files = glob.glob(os.path.join(self.plots_output_dir, "*_straddle_plot.png"))
            
            symbol_counts = {}
            for plot_file in plot_files:
                filename = os.path.basename(plot_file)
                symbol = filename.split('_')[0]
                
                if symbol not in symbol_counts:
                    symbol_counts[symbol] = 0
                symbol_counts[symbol] += 1
            
            # Print summary
            print(f"\n{'='*80}")
            print("STRADDLE PLOTS SUMMARY")
            print(f"{'='*80}")
            print(f"Total plots created: {successful_plots}")
            print(f"Failed plots: {failed_plots}")
            print(f"Plots saved to: {self.plots_output_dir}")
            print()
            
            if symbol_counts:
                print("PLOTS BY SYMBOL:")
                for symbol in sorted(symbol_counts.keys()):
                    print(f"  {symbol}: {symbol_counts[symbol]} plots")
            
            print(f"\n{'='*80}")
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")

def main():
    # Configuration
    straddle_data_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_data"
    plots_output_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_plots"
    
    # Create plotter
    plotter = StraddlePlotter(
        straddle_data_dir=straddle_data_dir,
        plots_output_dir=plots_output_dir
    )
    
    print("="*80)
    print("STRADDLE PLOTTER")
    print("="*80)
    print("Creating plots showing straddle price and IV over time for each contract...")
    print(f"Input directory: {straddle_data_dir}")
    print(f"Output directory: {plots_output_dir}")
    print()
    
    # Create all plots
    plotter.plot_all_straddles()

if __name__ == "__main__":
    main()