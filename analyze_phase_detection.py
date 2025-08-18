#!/usr/bin/env python3
"""
Phase Detection Analysis and Visualization Script

This script analyzes the gamestate data to visualize:
1. Phase detection parameters over time
2. Phase state transitions
3. Phase duration patterns
4. Data quality and consistency
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PhaseDetectionAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.gamestates_dir = self.data_dir / "gamestates"
        self.main_gamestate_file = self.data_dir / "runelite_gamestate.json"
        self.actions_file = self.data_dir / "actions.csv"
        
        # Phase detection parameters we want to track
        self.phase_params = [
            'bank_open', 'crafting_interface_open', 'has_materials',
            'player_animation', 'inventory_count', 'bank_count'
        ]
        
        self.data = []
        
    def load_gamestate_data(self):
        """Load and parse all gamestate files."""
        print("Loading gamestate data...")
        
        # Load main gamestate file
        if self.main_gamestate_file.exists():
            with open(self.main_gamestate_file, 'r') as f:
                main_data = json.load(f)
                self.data.append(main_data)
                print(f"Loaded main gamestate: {len(main_data)} fields")
        
        # Load individual gamestate files
        if self.gamestates_dir.exists():
            gamestate_files = sorted(list(self.gamestates_dir.glob("*.json")))
            print(f"Found {len(gamestate_files)} individual gamestate files")
            
            for i, file_path in enumerate(gamestate_files):
                try:
                    with open(file_path, 'r') as f:
                        gamestate = json.load(f)
                        self.data.append(gamestate)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Loaded {i + 1}/{len(gamestate_files)} gamestate files")
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Total gamestates loaded: {len(self.data)}")
        
    def extract_phase_parameters(self):
        """Extract phase detection parameters from gamestates."""
        print("Extracting phase detection parameters...")
        
        extracted_data = []
        
        for gamestate in self.data:
            try:
                # Basic info
                timestamp = gamestate.get('timestamp', 0)
                phase_context = gamestate.get('phase_context', {})
                
                # Phase detection parameters
                bank_open = gamestate.get('bank_open', False)
                
                # Check for crafting interface (widget-based detection)
                crafting_interface_open = False
                if 'crafting_interface' in gamestate:
                    crafting_interface_open = gamestate['crafting_interface']
                
                # Check for materials in inventory
                has_materials = False
                inventory = gamestate.get('inventory', [])
                if inventory:
                    for item in inventory:
                        if item.get('id') in [1607, 2357]:  # Sapphire or Gold bar
                            has_materials = True
                            break
                
                # Player animation
                player_animation = "unknown"
                if 'player' in gamestate:
                    player = gamestate['player']
                    animation_name = player.get('animation_name', 'unknown')
                    animation_id = player.get('animation_id', -1)
                    player_animation = f"{animation_name} ({animation_id})"
                
                # Inventory and bank counts
                inventory_count = len([item for item in inventory if item.get('id') != -1])
                bank_count = len(gamestate.get('bank', []))
                
                # Phase context
                cycle_phase = phase_context.get('cycle_phase', 'unknown')
                phase_start_time = phase_context.get('phase_start_time', 0)
                phase_duration_ms = phase_context.get('phase_duration_ms', 0)
                gamestates_in_phase = phase_context.get('gamestates_in_phase', 0)
                
                extracted_data.append({
                    'timestamp': timestamp,
                    'bank_open': bank_open,
                    'crafting_interface_open': crafting_interface_open,
                    'has_materials': has_materials,
                    'player_animation': player_animation,
                    'inventory_count': inventory_count,
                    'bank_count': bank_count,
                    'cycle_phase': cycle_phase,
                    'phase_start_time': phase_start_time,
                    'phase_duration_ms': phase_duration_ms,
                    'gamestates_in_phase': gamestates_in_phase
                })
                
            except Exception as e:
                print(f"Error extracting data from gamestate: {e}")
                continue
        
        self.extracted_df = pd.DataFrame(extracted_data)
        
        # Convert timestamps to datetime for better plotting
        if not self.extracted_df.empty:
            self.extracted_df['datetime'] = pd.to_datetime(self.extracted_df['timestamp'], unit='ms')
            self.extracted_df['time_seconds'] = (self.extracted_df['timestamp'] - self.extracted_df['timestamp'].min()) / 1000
            
        print(f"Extracted data for {len(self.extracted_df)} gamestates")
        
    def create_visualizations(self):
        """Create comprehensive visualizations of phase detection."""
        if self.extracted_df.empty:
            print("No data to visualize!")
            return
            
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Phase State Over Time
        ax1 = plt.subplot(4, 2, 1)
        self._plot_phase_over_time(ax1)
        
        # 2. Phase Detection Parameters Over Time
        ax2 = plt.subplot(4, 2, 2)
        self._plot_phase_parameters(ax2)
        
        # 3. Phase Duration Distribution
        ax3 = plt.subplot(4, 2, 3)
        self._plot_phase_duration_distribution(ax3)
        
        # 4. Gamestates per Phase
        ax4 = plt.subplot(4, 2, 4)
        self._plot_gamestates_per_phase(ax4)
        
        # 5. Phase Transition Matrix
        ax5 = plt.subplot(4, 2, 5)
        self._plot_phase_transitions(ax5)
        
        # 6. Parameter Correlation Heatmap
        ax6 = plt.subplot(4, 2, 6)
        self._plot_parameter_correlation(ax6)
        
        # 7. Phase Detection Logic Analysis
        ax7 = plt.subplot(4, 2, 7)
        self._plot_phase_logic_analysis(ax7)
        
        # 8. Data Quality Metrics
        ax8 = plt.subplot(4, 2, 8)
        self._plot_data_quality(ax8)
        
        plt.tight_layout()
        plt.savefig('phase_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed plots
        self._create_detailed_plots()
        
    def _plot_phase_over_time(self, ax):
        """Plot phase states over time."""
        phase_colors = {
            'banking': 'green',
            'crafting': 'blue', 
            'moving_to_furnace': 'orange',
            'moving_to_bank': 'red',
            'unknown': 'gray'
        }
        
        for phase in self.extracted_df['cycle_phase'].unique():
            if phase in phase_colors:
                phase_data = self.extracted_df[self.extracted_df['cycle_phase'] == phase]
                ax.scatter(phase_data['time_seconds'], [1] * len(phase_data), 
                          c=phase_colors[phase], label=phase, alpha=0.7, s=20)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Phase State')
        ax.set_title('Phase States Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_phase_parameters(self, ax):
        """Plot key phase detection parameters over time."""
        # Plot bank_open status
        ax.plot(self.extracted_df['time_seconds'], 
                self.extracted_df['bank_open'].astype(int), 
                label='Bank Open', linewidth=2, alpha=0.8)
        
        # Plot has_materials status
        ax.plot(self.extracted_df['time_seconds'], 
                self.extracted_df['has_materials'].astype(int) * 0.8, 
                label='Has Materials', linewidth=2, alpha=0.8)
        
        # Plot inventory count (normalized)
        inventory_normalized = self.extracted_df['inventory_count'] / self.extracted_df['inventory_count'].max()
        ax.plot(self.extracted_df['time_seconds'], inventory_normalized, 
                label='Inventory Count (norm)', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Phase Detection Parameters Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_phase_duration_distribution(self, ax):
        """Plot distribution of phase durations."""
        phase_durations = self.extracted_df.groupby('cycle_phase')['phase_duration_ms'].apply(list)
        
        for phase, durations in phase_durations.items():
            if durations and len(durations) > 1:
                # Convert to seconds
                durations_seconds = [d/1000 for d in durations if d > 0]
                if durations_seconds:
                    ax.hist(durations_seconds, alpha=0.7, label=phase, bins=20)
        
        ax.set_xlabel('Phase Duration (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Phase Duration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_gamestates_per_phase(self, ax):
        """Plot number of gamestates captured per phase."""
        gamestates_per_phase = self.extracted_df.groupby('cycle_phase')['gamestates_in_phase'].max()
        
        phases = list(gamestates_per_phase.index)
        counts = list(gamestates_per_phase.values)
        
        bars = ax.bar(phases, counts, alpha=0.7)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Max Gamestates in Phase')
        ax.set_title('Gamestates Captured per Phase')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        
    def _plot_phase_transitions(self, ax):
        """Plot phase transition matrix."""
        # Create phase transition matrix
        phases = self.extracted_df['cycle_phase'].unique()
        transition_matrix = pd.DataFrame(0, index=phases, columns=phases)
        
        for i in range(len(self.extracted_df) - 1):
            current_phase = self.extracted_df.iloc[i]['cycle_phase']
            next_phase = self.extracted_df.iloc[i + 1]['cycle_phase']
            if current_phase in phases and next_phase in phases:
                transition_matrix.loc[current_phase, next_phase] += 1
        
        # Plot heatmap
        sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('Phase Transition Matrix')
        ax.set_xlabel('Next Phase')
        ax.set_ylabel('Current Phase')
        
    def _plot_parameter_correlation(self, ax):
        """Plot correlation between phase detection parameters."""
        # Select numeric columns for correlation
        numeric_cols = ['bank_open', 'has_materials', 'inventory_count', 'bank_count']
        correlation_data = self.extracted_df[numeric_cols].astype(float)
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Parameter Correlation Matrix')
        
    def _plot_phase_logic_analysis(self, ax):
        """Analyze how well the phase detection logic is working."""
        # Check if phase detection matches expected logic
        logic_analysis = []
        
        for _, row in self.extracted_df.iterrows():
            bank_open = row['bank_open']
            has_materials = row['has_materials']
            detected_phase = row['cycle_phase']
            
            # Expected phase based on logic
            if bank_open:
                expected_phase = 'banking'
            elif has_materials:
                expected_phase = 'moving_to_furnace'
            else:
                expected_phase = 'moving_to_bank'
            
            logic_analysis.append({
                'detected': detected_phase,
                'expected': expected_phase,
                'matches': detected_phase == expected_phase
            })
        
        logic_df = pd.DataFrame(logic_analysis)
        match_rate = logic_df['matches'].mean() * 100
        
        # Create a simple visualization
        ax.text(0.5, 0.6, f'Phase Detection Accuracy: {match_rate:.1f}%', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        # Show breakdown
        phase_accuracy = logic_df.groupby('detected')['matches'].mean()
        for i, (phase, accuracy) in enumerate(phase_accuracy.items()):
            y_pos = 0.4 - (i * 0.1)
            ax.text(0.5, y_pos, f'{phase}: {accuracy*100:.1f}%', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Phase Detection Logic Analysis')
        ax.axis('off')
        
    def _plot_data_quality(self, ax):
        """Plot data quality metrics."""
        # Calculate data quality metrics
        total_gamestates = len(self.extracted_df)
        phases_with_data = self.extracted_df['cycle_phase'].nunique()
        avg_phase_duration = self.extracted_df['phase_duration_ms'].mean() / 1000
        data_completeness = self.extracted_df.notna().mean().mean() * 100
        
        metrics = [
            f'Total Gamestates: {total_gamestates}',
            f'Unique Phases: {phases_with_data}',
            f'Avg Phase Duration: {avg_phase_duration:.1f}s',
            f'Data Completeness: {data_completeness:.1f}%'
        ]
        
        for i, metric in enumerate(metrics):
            y_pos = 0.8 - (i * 0.2)
            ax.text(0.5, y_pos, metric, ha='center', va='center', 
                   fontsize=12, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Data Quality Metrics')
        ax.axis('off')
        
    def _create_detailed_plots(self):
        """Create additional detailed plots."""
        print("Creating detailed plots...")
        
        # 1. Phase Duration Timeline
        plt.figure(figsize=(15, 8))
        self._plot_phase_duration_timeline()
        
        # 2. Parameter State Analysis
        plt.figure(figsize=(15, 8))
        self._plot_parameter_state_analysis()
        
        # 3. Phase Detection Debug
        plt.figure(figsize=(15, 8))
        self._plot_phase_detection_debug()
        
    def _plot_phase_duration_timeline(self):
        """Plot phase durations as a timeline."""
        # Group by phase and show duration over time
        for phase in self.extracted_df['cycle_phase'].unique():
            phase_data = self.extracted_df[self.extracted_df['cycle_phase'] == phase]
            if not phase_data.empty:
                plt.scatter(phase_data['time_seconds'], 
                           phase_data['phase_duration_ms'] / 1000,
                           label=phase, alpha=0.7, s=30)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Phase Duration (seconds)')
        plt.title('Phase Duration Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('phase_duration_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_parameter_state_analysis(self):
        """Analyze the state of parameters during each phase."""
        # Create subplots for each parameter
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter State Analysis by Phase', fontsize=16)
        
        # Bank open status by phase
        bank_by_phase = self.extracted_df.groupby('cycle_phase')['bank_open'].value_counts()
        bank_by_phase.unstack().plot(kind='bar', ax=axes[0,0], title='Bank Status by Phase')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Materials status by phase
        materials_by_phase = self.extracted_df.groupby('cycle_phase')['has_materials'].value_counts()
        materials_by_phase.unstack().plot(kind='bar', ax=axes[0,1], title='Materials Status by Phase')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Inventory count by phase
        self.extracted_df.boxplot(column='inventory_count', by='cycle_phase', ax=axes[1,0])
        axes[1,0].set_title('Inventory Count by Phase')
        axes[1,0].set_xlabel('Phase')
        axes[1,0].set_ylabel('Item Count')
        
        # Player animation by phase
        animation_by_phase = self.extracted_df.groupby('cycle_phase')['player_animation'].value_counts()
        animation_by_phase.unstack().plot(kind='bar', ax=axes[1,1], title='Player Animation by Phase')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('parameter_state_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_phase_detection_debug(self):
        """Debug view of phase detection logic."""
        # Show a few examples of phase detection
        debug_data = self.extracted_df[['timestamp', 'bank_open', 'has_materials', 
                                       'cycle_phase', 'phase_duration_ms']].head(20)
        
        # Create a table visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = []
        for _, row in debug_data.iterrows():
            table_data.append([
                datetime.fromtimestamp(row['timestamp']/1000).strftime('%H:%M:%S'),
                str(row['bank_open']),
                str(row['has_materials']),
                row['cycle_phase'],
                f"{row['phase_duration_ms']/1000:.1f}s"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Time', 'Bank Open', 'Has Materials', 'Detected Phase', 'Duration'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Phase Detection Debug View (First 20 Gamestates)', fontsize=14)
        plt.savefig('phase_detection_debug.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if self.extracted_df.empty:
            print("No data to analyze!")
            return
            
        print("\n" + "="*60)
        print("PHASE DETECTION ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total Gamestates: {len(self.extracted_df)}")
        print(f"   Time Range: {self.extracted_df['time_seconds'].min():.1f}s - {self.extracted_df['time_seconds'].max():.1f}s")
        print(f"   Total Duration: {self.extracted_df['time_seconds'].max() - self.extracted_df['time_seconds'].min():.1f}s")
        
        # Phase analysis
        print(f"\n🔄 PHASE ANALYSIS:")
        phase_counts = self.extracted_df['cycle_phase'].value_counts()
        for phase, count in phase_counts.items():
            percentage = (count / len(self.extracted_df)) * 100
            print(f"   {phase}: {count} gamestates ({percentage:.1f}%)")
        
        # Phase duration analysis
        print(f"\n⏱️  PHASE DURATION ANALYSIS:")
        phase_durations = self.extracted_df.groupby('cycle_phase')['phase_duration_ms'].agg(['mean', 'min', 'max'])
        for phase in phase_durations.index:
            mean_dur = phase_durations.loc[phase, 'mean'] / 1000
            min_dur = phase_durations.loc[phase, 'min'] / 1000
            max_dur = phase_durations.loc[phase, 'max'] / 1000
            print(f"   {phase}: avg={mean_dur:.1f}s, min={min_dur:.1f}s, max={max_dur:.1f}s")
        
        # Data quality
        print(f"\n🔍 DATA QUALITY:")
        missing_data = self.extracted_df.isnull().sum()
        if missing_data.sum() > 0:
            print("   Missing data detected:")
            for col, missing in missing_data.items():
                if missing > 0:
                    print(f"     {col}: {missing} missing values")
        else:
            print("   ✅ No missing data detected")
        
        # Phase detection logic validation
        print(f"\n🧠 PHASE DETECTION LOGIC VALIDATION:")
        self._validate_phase_detection_logic()
        
        print("\n" + "="*60)
        
    def _validate_phase_detection_logic(self):
        """Validate if phase detection logic is working correctly."""
        logic_errors = []
        total_checks = 0
        
        for _, row in self.extracted_df.iterrows():
            total_checks += 1
            bank_open = row['bank_open']
            has_materials = row['has_materials']
            detected_phase = row['cycle_phase']
            
            # Expected logic
            if bank_open and detected_phase != 'banking':
                logic_errors.append(f"Bank open but phase is {detected_phase}")
            elif not bank_open and has_materials and detected_phase != 'moving_to_furnace':
                logic_errors.append(f"Has materials, bank closed, but phase is {detected_phase}")
            elif not bank_open and not has_materials and detected_phase != 'moving_to_bank':
                logic_errors.append(f"No materials, bank closed, but phase is {detected_phase}")
        
        if logic_errors:
            print(f"   ⚠️  {len(logic_errors)} logic errors detected:")
            for error in logic_errors[:5]:  # Show first 5 errors
                print(f"     - {error}")
            if len(logic_errors) > 5:
                print(f"     ... and {len(logic_errors) - 5} more errors")
        else:
            print("   ✅ Phase detection logic is working correctly")
        
        print(f"   Logic validation: {total_checks - len(logic_errors)}/{total_checks} correct ({((total_checks - len(logic_errors))/total_checks)*100:.1f}%)")

def main():
    """Main function to run the phase detection analysis."""
    print("🔍 Phase Detection Analysis Script")
    print("="*50)
    
    # Initialize analyzer
    analyzer = PhaseDetectionAnalyzer()
    
    # Load data
    analyzer.load_gamestate_data()
    
    # Extract parameters
    analyzer.extract_phase_parameters()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print("\n✅ Analysis complete! Check the generated PNG files for visualizations.")
    print("📊 Summary report has been printed above.")

if __name__ == "__main__":
    main()
