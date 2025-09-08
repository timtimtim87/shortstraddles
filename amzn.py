#!/usr/bin/env python3
"""
Straddle Price Change Attribution Analysis - Rolling 7-Day with Weighted Attribution

This script analyzes how much each leg (call vs put) contributes to straddle price changes
using a 7-day rolling window and weights based on each leg's contribution to total straddle value.
Attribution scale: -1 = all change from call, +1 = all change from put, 0 = equal contribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def load_and_process_data(file_path, rolling_days=7):
    """Load CSV data and calculate rolling attribution metrics."""
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Find the call and put columns dynamically
    call_col = None
    put_col = None
    
    for col in df.columns:
        if 'C250.0' in col and 'close' in col:
            call_col = col
        elif 'P250.0' in col and 'close' in col:
            put_col = col
    
    if not call_col or not put_col:
        raise ValueError("Could not find call or put price columns in the data")
    
    print(f"Found call column: {call_col}")
    print(f"Found put column: {put_col}")
    
    # Calculate rolling changes
    df[f'straddle_change_{rolling_days}d'] = df['close'].diff(rolling_days)
    df[f'call_change_{rolling_days}d'] = df[call_col].diff(rolling_days)
    df[f'put_change_{rolling_days}d'] = df[put_col].diff(rolling_days)
    
    # Calculate daily changes for reference
    df['straddle_change_1d'] = df['close'].diff()
    df['call_change_1d'] = df[call_col].diff()
    df['put_change_1d'] = df[put_col].diff()
    
    # Calculate weights based on each leg's contribution to total straddle value
    df['call_weight'] = df[call_col] / df['close']
    df['put_weight'] = df[put_col] / df['close']
    
    # Calculate rolling averages for smoother analysis
    df['call_weight_rolling'] = df['call_weight'].rolling(window=rolling_days, center=True).mean()
    df['put_weight_rolling'] = df['put_weight'].rolling(window=rolling_days, center=True).mean()
    
    # Method 1: Simple attribution based on rolling changes
    attribution_simple = []
    
    # Method 2: Weighted attribution considering each leg's value contribution
    attribution_weighted = []
    
    # Method 3: Hybrid approach - combines change attribution with value weighting
    attribution_hybrid = []
    
    for i in range(len(df)):
        straddle_chg = df.loc[i, f'straddle_change_{rolling_days}d']
        call_chg = df.loc[i, f'call_change_{rolling_days}d']
        put_chg = df.loc[i, f'put_change_{rolling_days}d']
        
        call_weight = df.loc[i, 'call_weight_rolling']
        put_weight = df.loc[i, 'put_weight_rolling']
        
        # Skip if we don't have enough data or weights
        if pd.isna(straddle_chg) or pd.isna(call_chg) or pd.isna(put_chg) or pd.isna(call_weight):
            attribution_simple.append(np.nan)
            attribution_weighted.append(np.nan)
            attribution_hybrid.append(np.nan)
            continue
        
        # Method 1: Simple attribution (same as before but with rolling changes)
        total_component_change = abs(call_chg) + abs(put_chg)
        if total_component_change < 0.001:
            attr_simple = 0.0
        else:
            attr_simple = (put_chg - call_chg) / total_component_change
        
        # Method 2: Weighted attribution based on value contribution
        # If put is worth more in the straddle, its changes should matter more
        if abs(straddle_chg) < 0.001:
            attr_weighted = 0.0
        else:
            # Weight the changes by how much each leg contributes to total value
            weighted_call_impact = call_chg * call_weight
            weighted_put_impact = put_chg * put_weight
            
            total_weighted_impact = abs(weighted_call_impact) + abs(weighted_put_impact)
            if total_weighted_impact < 0.001:
                attr_weighted = 0.0
            else:
                attr_weighted = (weighted_put_impact - weighted_call_impact) / total_weighted_impact
        
        # Method 3: Hybrid approach - blend change attribution with value weighting
        # This gives more influence to the leg that's worth more
        if total_component_change < 0.001:
            attr_hybrid = 0.0
        else:
            # Base attribution from changes
            base_attribution = (put_chg - call_chg) / total_component_change
            
            # Modify attribution based on which leg is worth more
            # If put is worth 80% of straddle, put changes should be weighted more heavily
            weight_bias = put_weight - call_weight  # Range: -1 to +1
            
            # Combine base attribution with weight bias (50/50 blend)
            attr_hybrid = 0.7 * base_attribution + 0.3 * weight_bias
            
        attribution_simple.append(attr_simple)
        attribution_weighted.append(attr_weighted)
        attribution_hybrid.append(attr_hybrid)
    
    df['attribution_simple'] = attribution_simple
    df['attribution_weighted'] = attribution_weighted  
    df['attribution_hybrid'] = attribution_hybrid
    
    # Add some additional useful columns
    df['theoretical_straddle_change'] = df[f'call_change_{rolling_days}d'] + df[f'put_change_{rolling_days}d']
    
    # Store column names for later use
    df.attrs['call_col'] = call_col
    df.attrs['put_col'] = put_col
    df.attrs['rolling_days'] = rolling_days
    
    return df

def create_plots(df):
    """Create the attribution and price plots."""
    
    rolling_days = df.attrs['rolling_days']
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Straddle Price Change Attribution Analysis - {rolling_days}-Day Rolling\n'
                f'AMZN Jan 16, 2026 $250 Straddle', 
                 fontsize=16, fontweight='bold')
    
    # Remove NaN values for plotting
    plot_df = df.dropna()
    
    # Plot 1: All three attribution methods
    ax1 = axes[0]
    ax1.plot(plot_df['datetime'], plot_df['attribution_simple'], 
             color='steelblue', linewidth=2, label='Simple Attribution', alpha=0.7)
    ax1.plot(plot_df['datetime'], plot_df['attribution_weighted'], 
             color='red', linewidth=2, label='Weighted Attribution', alpha=0.8)
    ax1.plot(plot_df['datetime'], plot_df['attribution_hybrid'], 
             color='darkgreen', linewidth=3, label='Hybrid Attribution')
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax1.set_ylabel('Attribution Score', fontweight='bold')
    ax1.set_title(f'{rolling_days}-Day Rolling Attribution: -1 = Call Driven, +1 = Put Driven')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # Plot 2: Value weights over time
    ax2 = axes[1]
    ax2.plot(plot_df['datetime'], plot_df['call_weight_rolling'], 
             color='red', linewidth=2, label='Call Weight', alpha=0.8)
    ax2.plot(plot_df['datetime'], plot_df['put_weight_rolling'], 
             color='green', linewidth=2, label='Put Weight', alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_ylabel('Value Weight', fontweight='bold')
    ax2.set_title('Each Leg\'s Contribution to Total Straddle Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Component Prices
    ax3 = axes[2]
    ax3.plot(plot_df['datetime'], plot_df['close'], 
             color='purple', linewidth=2, label='Straddle Price', marker='o', markersize=3)
    ax3.plot(plot_df['datetime'], plot_df[df.attrs['call_col']], 
             color='red', linewidth=2, label='Call Price', alpha=0.8)
    ax3.plot(plot_df['datetime'], plot_df[df.attrs['put_col']], 
             color='green', linewidth=2, label='Put Price', alpha=0.8)
    
    ax3.set_ylabel('Price ($)', fontweight='bold')
    ax3.set_title('Component Prices Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling Price Changes
    ax4 = axes[3]
    width = 0.8
    ax4.bar(plot_df['datetime'], plot_df[f'straddle_change_{rolling_days}d'], 
            color='purple', alpha=0.6, label=f'Straddle Change ({rolling_days}d)', width=width)
    ax4.bar(plot_df['datetime'], plot_df[f'call_change_{rolling_days}d'], 
            color='red', alpha=0.5, label=f'Call Change ({rolling_days}d)', width=width*0.6)
    ax4.bar(plot_df['datetime'], plot_df[f'put_change_{rolling_days}d'], 
            color='green', alpha=0.5, label=f'Put Change ({rolling_days}d)', width=width*0.6)
    
    ax4.set_ylabel('Price Change ($)', fontweight='bold')
    ax4.set_title(f'{rolling_days}-Day Rolling Price Changes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    
    # Plot 5: Attribution vs Value Weight Relationship
    ax5 = axes[4]
    # Scatter plot showing relationship between put weight and attribution
    valid_data = plot_df.dropna(subset=['put_weight_rolling', 'attribution_hybrid'])
    scatter = ax5.scatter(valid_data['put_weight_rolling'], valid_data['attribution_hybrid'], 
                         c=valid_data.index, cmap='viridis', alpha=0.6, s=30)
    
    ax5.set_xlabel('Put Weight (fraction of total straddle value)', fontweight='bold')
    ax5.set_ylabel('Hybrid Attribution Score', fontweight='bold')
    ax5.set_title('Attribution Score vs Put Value Weight')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax5.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add colorbar for time progression
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Time Progression', rotation=270, labelpad=15)
    
    # Format x-axis for all subplots except the scatter plot
    for ax in axes[:-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[-2].set_xlabel('Date', fontweight='bold')
    
    plt.tight_layout()
    return fig

def print_summary_stats(df):
    """Print summary statistics."""
    
    rolling_days = df.attrs['rolling_days']
    
    # Remove NaN values
    clean_df = df.dropna()
    
    print("\n" + "="*70)
    print(f"STRADDLE ATTRIBUTION ANALYSIS SUMMARY ({rolling_days}-DAY ROLLING)")
    print("="*70)
    
    print(f"Analysis Period: {clean_df['datetime'].min().strftime('%Y-%m-%d')} to {clean_df['datetime'].max().strftime('%Y-%m-%d')}")
    print(f"Total Trading Days: {len(clean_df)}")
    
    # Compare all three attribution methods
    methods = {
        'Simple': 'attribution_simple',
        'Weighted': 'attribution_weighted', 
        'Hybrid': 'attribution_hybrid'
    }
    
    print(f"\nATTRIBUTION METHOD COMPARISON:")
    print(f"{'Method':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Call%':<8} {'Put%':<8}")
    print("-" * 65)
    
    for method_name, col_name in methods.items():
        if col_name in clean_df.columns:
            data = clean_df[col_name]
            call_dom = (data < -0.5).sum() / len(data) * 100
            put_dom = (data > 0.5).sum() / len(data) * 100
            
            print(f"{method_name:<12} {data.mean():<8.3f} {data.std():<8.3f} "
                  f"{data.min():<8.3f} {data.max():<8.3f} {call_dom:<8.1f} {put_dom:<8.1f}")
    
    # Value weight statistics
    print(f"\nVALUE WEIGHT STATISTICS:")
    avg_call_weight = clean_df['call_weight_rolling'].mean()
    avg_put_weight = clean_df['put_weight_rolling'].mean()
    print(f"  Average Call Weight: {avg_call_weight:.3f} ({avg_call_weight*100:.1f}%)")
    print(f"  Average Put Weight:  {avg_put_weight:.3f} ({avg_put_weight*100:.1f}%)")
    
    # Price change statistics
    print(f"\n{rolling_days}-DAY ROLLING PRICE CHANGE STATISTICS:")
    print(f"  Mean Straddle Change: ${clean_df[f'straddle_change_{rolling_days}d'].mean():.3f}")
    print(f"  Mean Call Change:     ${clean_df[f'call_change_{rolling_days}d'].mean():.3f}")
    print(f"  Mean Put Change:      ${clean_df[f'put_change_{rolling_days}d'].mean():.3f}")
    
    print(f"\n{rolling_days}-DAY VOLATILITY (Std Dev of Changes):")
    print(f"  Straddle: ${clean_df[f'straddle_change_{rolling_days}d'].std():.3f}")
    print(f"  Call:     ${clean_df[f'call_change_{rolling_days}d'].std():.3f}")
    print(f"  Put:      ${clean_df[f'put_change_{rolling_days}d'].std():.3f}")
    
    # Correlation analysis
    if len(clean_df) > 1:
        print(f"\nCORRELATION ANALYSIS:")
        corr_put_weight_attr = clean_df['put_weight_rolling'].corr(clean_df['attribution_hybrid'])
        print(f"  Put Weight vs Hybrid Attribution: {corr_put_weight_attr:.3f}")

def main():
    """Main function to run the analysis."""
    
    # File path - modify this to match your file location
    file_path = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/OPRA_DLY_AMZN260116C250.0+OPRA_DLY_AMZN260116P250.0, 1D.csv"
    
    # Rolling window size
    rolling_days = 7
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please update the file_path variable in the script to match your file location.")
        return
    
    try:
        # Load and process the data
        print(f"Loading and processing data with {rolling_days}-day rolling window...")
        df = load_and_process_data(file_path, rolling_days)
        
        # Print summary statistics
        print_summary_stats(df)
        
        # Create plots
        print("\nCreating plots...")
        fig = create_plots(df)
        
        # Show the plot
        plt.show()
        
        # Optionally save the plot
        save_plot = input("\nSave plot to file? (y/n): ").lower().strip()
        if save_plot == 'y':
            output_file = f"straddle_attribution_analysis_{rolling_days}day.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {output_file}")
        
        # Optionally save processed data
        save_data = input("Save processed data to CSV? (y/n): ").lower().strip()
        if save_data == 'y':
            output_csv = f"straddle_attribution_data_{rolling_days}day.csv"
            df.to_csv(output_csv, index=False)
            print(f"Processed data saved as {output_csv}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file path and data format.")

if __name__ == "__main__":
    main()