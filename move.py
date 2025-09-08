import os
import shutil
import glob
import logging
from datetime import datetime

class StraddlePlotOrganizer:
    def __init__(self, plots_dir: str):
        """
        Initialize the plot organizer.
        
        Args:
            plots_dir: Directory containing straddle plot files
        """
        self.plots_dir = plots_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.move_stats = {
            '2025': 0,
            '2026': 0,
            'other': 0,
            'failed': 0
        }
    
    def extract_expiry_year(self, filename: str) -> str:
        """Extract expiry year from filename."""
        try:
            # Example filename: AAPL_2025-01-17_210_straddle_plot.png
            parts = filename.split('_')
            if len(parts) >= 2:
                expiry_date = parts[1]  # Should be like "2025-01-17"
                year = expiry_date.split('-')[0]  # Extract "2025"
                return year
        except Exception as e:
            self.logger.error(f"Error extracting year from {filename}: {e}")
        
        return None
    
    def create_year_folders(self):
        """Create folders for each expiry year."""
        years = ['2025', '2026']
        
        for year in years:
            year_folder = os.path.join(self.plots_dir, year)
            os.makedirs(year_folder, exist_ok=True)
            self.logger.info(f"Created/verified folder: {year_folder}")
    
    def organize_plots(self, dry_run: bool = True):
        """Organize plots into year-based folders."""
        self.logger.info(f"Starting plot organization {'(DRY RUN)' if dry_run else '(LIVE RUN)'}")
        
        # Find all straddle plot files
        pattern = os.path.join(self.plots_dir, "*_straddle_plot.png")
        plot_files = glob.glob(pattern)
        
        if not plot_files:
            self.logger.error(f"No straddle plot files found in {self.plots_dir}")
            return
        
        self.logger.info(f"Found {len(plot_files)} plot files to organize")
        
        # Create year folders if not dry run
        if not dry_run:
            self.create_year_folders()
        
        # Process each file
        for plot_path in plot_files:
            filename = os.path.basename(plot_path)
            
            try:
                # Extract expiry year
                year = self.extract_expiry_year(filename)
                
                if year in ['2025', '2026']:
                    # Determine destination
                    dest_folder = os.path.join(self.plots_dir, year)
                    dest_path = os.path.join(dest_folder, filename)
                    
                    if dry_run:
                        self.logger.info(f"WOULD MOVE: {filename} -> {year}/")
                        self.move_stats[year] += 1
                    else:
                        # Move the file
                        shutil.move(plot_path, dest_path)
                        self.logger.info(f"MOVED: {filename} -> {year}/")
                        self.move_stats[year] += 1
                        
                else:
                    # Unknown or unsupported year
                    self.logger.warning(f"Unknown expiry year for file: {filename}")
                    self.move_stats['other'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
                self.move_stats['failed'] += 1
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print organization summary."""
        total_processed = sum(self.move_stats.values())
        
        print(f"\n{'='*60}")
        print("PLOT ORGANIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Files moved to 2025 folder: {self.move_stats['2025']}")
        print(f"Files moved to 2026 folder: {self.move_stats['2026']}")
        print(f"Files with unknown years: {self.move_stats['other']}")
        print(f"Failed to process: {self.move_stats['failed']}")
        print(f"Total files processed: {total_processed}")
        print()
        
        if self.move_stats['2025'] > 0:
            print(f"2025 plots location: {os.path.join(self.plots_dir, '2025')}")
        if self.move_stats['2026'] > 0:
            print(f"2026 plots location: {os.path.join(self.plots_dir, '2026')}")
        
        print(f"{'='*60}")
    
    def verify_organization(self):
        """Verify the organization by counting files in each folder."""
        self.logger.info("Verifying organization...")
        
        # Count files in main directory
        main_plots = glob.glob(os.path.join(self.plots_dir, "*_straddle_plot.png"))
        
        # Count files in year folders
        folder_2025 = os.path.join(self.plots_dir, '2025')
        folder_2026 = os.path.join(self.plots_dir, '2026')
        
        plots_2025 = glob.glob(os.path.join(folder_2025, "*_straddle_plot.png")) if os.path.exists(folder_2025) else []
        plots_2026 = glob.glob(os.path.join(folder_2026, "*_straddle_plot.png")) if os.path.exists(folder_2026) else []
        
        print(f"\nVERIFICATION RESULTS:")
        print(f"Plots remaining in main directory: {len(main_plots)}")
        print(f"Plots in 2025 folder: {len(plots_2025)}")
        print(f"Plots in 2026 folder: {len(plots_2026)}")
        
        if len(main_plots) == 0:
            print("✓ All plots have been organized into year folders")
        else:
            print("⚠ Some plots remain in the main directory")
            
            # Show a few examples
            if main_plots:
                print("Examples of remaining files:")
                for plot in main_plots[:3]:
                    print(f"  {os.path.basename(plot)}")
    
    def run_organization(self):
        """Run the complete organization process."""
        print("="*60)
        print("STRADDLE PLOT ORGANIZER")
        print("="*60)
        print("This will organize your straddle plots into folders by expiry year:")
        print("- 2025 expiry plots -> 2025/ folder")
        print("- 2026 expiry plots -> 2026/ folder")
        print(f"Working directory: {self.plots_dir}")
        print()
        
        # First run dry run to show what would happen
        print("Analyzing files...")
        self.organize_plots(dry_run=True)
        
        # Ask for confirmation
        response = input("\nProceed with organizing plots? (type 'yes' to confirm): ")
        
        if response.lower() == 'yes':
            print("Organizing plots...")
            # Reset stats for actual run
            self.move_stats = {'2025': 0, '2026': 0, 'other': 0, 'failed': 0}
            self.organize_plots(dry_run=False)
            
            # Verify results
            self.verify_organization()
            print("Organization completed!")
        else:
            print("Organization cancelled.")

def main():
    # Configuration - update this path to your straddle plots directory
    plots_dir = "/Users/tim/CODE_PROJECTS/SHORT_STRADDLES/straddle_plots"
    
    # Create organizer
    organizer = StraddlePlotOrganizer(plots_dir)
    
    # Run organization
    organizer.run_organization()

if __name__ == "__main__":
    main()