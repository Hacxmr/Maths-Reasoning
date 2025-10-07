"""
This script generates a dataset of 100 sed puzzles with varying difficulty levels.
The puzzles are saved in JSON format in the dataset/puzzles directory.
"""
import sys
from pathlib import Path

# Import the PuzzleGenerator class
from src.generator import PuzzleGenerator

if __name__ == "__main__":
    # Create a generator with default settings
    generator = PuzzleGenerator(
        min_string_length=2,
        max_string_length=25,
        min_transitions=2,
        max_transitions=12
    )
    
    # Define a custom difficulty distribution to ensure good coverage
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
    
    # Save the puzzles
    output_dir = Path("dataset/puzzles")
    generator.save_puzzles(puzzles, output_dir)
    
    print(f"Successfully generated 100 puzzles in {output_dir}")