"""
Script to generate 100 puzzles for the SED Puzzle dataset.
"""

import os
import sys
from pathlib import Path

# Get the current script's directory
current_dir = Path(__file__).resolve().parent

# Run the generator directly with a larger number of puzzles
if __name__ == "__main__":
    # Add the src directory to the Python path
    sys.path.append(str(current_dir / "src"))
    
    # Import after adjusting path
    from generator import PuzzleGenerator

    # Create output directory if it doesn't exist
    output_dir = current_dir / "dataset" / "puzzles"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = PuzzleGenerator(
        min_string_length=2,
        max_string_length=25,
        min_transitions=2,
        max_transitions=12
    )
    
    # Define difficulty distribution
    difficulty_distribution = {
        1: 0.05,  # Very easy (5%)
        2: 0.10,  # Easy (10%)
        3: 0.15,  # Easy-medium (15%) 
        4: 0.20,  # Medium (20%)
        5: 0.20,  # Medium (20%)
        6: 0.15,  # Medium-hard (15%)
        7: 0.10,  # Hard (10%)
        8: 0.05   # Very hard (5%)
    }
    
    # Generate 100 puzzles
    print("Generating 100 puzzles with varying difficulty levels...")
    puzzles = generator.generate_puzzles(
        num_puzzles=100,
        difficulty_distribution=difficulty_distribution
    )
    
    # Save puzzles
    generator.save_puzzles(puzzles, output_dir)
    
    print(f"Successfully generated 100 puzzles in {output_dir}")