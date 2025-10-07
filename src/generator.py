import json
import logging
import os
import random
import string
import sys
import time
from pathlib import Path

# Add the parent directory to the path if running as main script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from src.baseline import bfs
    from src.schema import Problem, Solution, Transition
else:
    from baseline import bfs
    from schema import Problem, Solution, Transition

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

class PuzzleGenerator:
    """
    A class to generate sed puzzles with varying difficulty levels.
    """
    
    def __init__(self, 
                 min_string_length=3, 
                 max_string_length=20,
                 min_transitions=2, 
                 max_transitions=10,
                 characters=string.ascii_letters + string.digits + " "):
        """
        Initialize the puzzle generator with configuration parameters.
        
        Args:
            min_string_length: Minimum length of the initial string
            max_string_length: Maximum length of the initial string
            min_transitions: Minimum number of transitions in a puzzle
            max_transitions: Maximum number of transitions in a puzzle
            characters: Set of characters to use in generating strings
        """
        self.min_string_length = min_string_length
        self.max_string_length = max_string_length
        self.min_transitions = min_transitions
        self.max_transitions = max_transitions
        self.characters = characters
        
    def _generate_random_string(self, min_length, max_length):
        """Generate a random string with length between min_length and max_length."""
        length = random.randint(min_length, max_length)
        return ''.join(random.choice(self.characters) for _ in range(length))
    
    def _generate_random_transition_pattern(self, source_str, min_length=1, max_length=None):
        """
        Generate a random substring from the source string to use as a transition pattern.
        
        Args:
            source_str: The string from which to extract a pattern
            min_length: Minimum length of the pattern
            max_length: Maximum length of the pattern (defaults to half of source string length)
        """
        if max_length is None:
            max_length = max(min_length, len(source_str) // 2)
        
        max_length = min(max_length, len(source_str))
        
        if min_length > max_length:
            min_length = max_length
            
        # If the string is too short, just return it
        if len(source_str) <= min_length:
            return source_str
            
        pattern_length = random.randint(min_length, max_length)
        
        # Pick a random starting position
        if len(source_str) - pattern_length + 1 <= 0:
            start_pos = 0
        else:
            start_pos = random.randint(0, len(source_str) - pattern_length)
            
        return source_str[start_pos:start_pos + pattern_length]
    
    def generate_forward_puzzle(self, difficulty):
        """
        Generate a puzzle by starting with a final state (empty string) and working backwards.
        This ensures the puzzle is solvable.
        
        Args:
            difficulty: Integer from 1-10 indicating puzzle difficulty
        
        Returns:
            Problem: A generated puzzle that is guaranteed to be solvable
        """
        # Start with empty string (final state)
        current_string = ""
        
        # Determine number of transitions based on difficulty
        num_transitions = min(self.min_transitions + difficulty, self.max_transitions)
        
        # Create transitions by applying them in reverse
        transitions = []
        transition_records = []  # To keep track of what transitions we used (for solution)
        
        # Calculate string complexity based on difficulty
        string_complexity = min(self.min_string_length + difficulty, self.max_string_length)
        
        for i in range(num_transitions):
            # Generate a target string that will be inserted
            if i == 0:
                # First transition should create a string from empty
                target_string = self._generate_random_string(
                    min(2, string_complexity), 
                    min(5, string_complexity)
                )
                source_string = ""  # Empty string as source
            else:
                # For subsequent transitions, insert or replace strings
                # Generate a non-empty target string
                target_length = max(1, min(3, string_complexity - len(current_string)))
                target_string = self._generate_random_string(1, target_length)
                
                # For source string, either use an empty string (insertion)
                # or extract a pattern from the current string (replacement)
                use_empty_source = random.random() < 0.3  # 30% chance to use empty source
                
                if use_empty_source or not current_string:
                    source_string = ""
                else:
                    # Extract a pattern from current string
                    source_string = self._generate_random_transition_pattern(
                        current_string, 
                        min_length=1,
                        max_length=min(len(current_string), 3)
                    )
            
            # Apply the transition in reverse
            if source_string:
                # Find a position where source_string exists in current_string
                if source_string in current_string:
                    pos = current_string.find(source_string)
                    current_string = current_string[:pos] + target_string + current_string[pos+len(source_string):]
                    position = pos  # Record where we made the change
                else:
                    # If we can't find the source pattern, insert at a random position
                    pos = random.randint(0, len(current_string))
                    current_string = current_string[:pos] + target_string + current_string[pos:]
                    source_string = ""  # Reset to empty string since we're doing an insertion
                    position = pos
            else:
                # For empty source string, insert at a random position
                pos = random.randint(0, len(current_string))
                current_string = current_string[:pos] + target_string + current_string[pos:]
                position = pos
            
            # Record the transition (reversed for the actual puzzle)
            transitions.append(Transition(src=target_string, tgt=source_string))
            transition_records.append((position, len(target_string), len(source_string)))
        
        # Create and return the puzzle
        problem_id = f"{random.randint(1000, 9999)}"
        
        # Ensure we have at least one transition with empty target (requirement)
        has_empty_target = any(t.tgt == "" for t in transitions)
        if not has_empty_target:
            # Replace a random transition with one that has an empty target
            rand_index = random.randint(0, len(transitions) - 1)
            transitions[rand_index] = Transition(src=transitions[rand_index].src, tgt="")
        
        # Create the puzzle
        puzzle = Problem(
            problem_id=problem_id,
            initial_string=current_string,
            transitions=transitions
        )
        
        # Verify the puzzle is solvable
        solution = bfs(puzzle, time_limit=10)
        if solution is None:
            # If puzzle is not solvable within time limit, generate a simpler one
            return self.generate_forward_puzzle(max(1, difficulty - 1))
            
        return puzzle
        
    def generate_puzzles(self, num_puzzles=100, difficulty_distribution=None):
        """
        Generate a set of puzzles with varying difficulty levels.
        
        Args:
            num_puzzles: Number of puzzles to generate
            difficulty_distribution: Optional distribution of difficulties (1-10)
                                     If None, will use a bell curve distribution
        
        Returns:
            dict: Dictionary mapping problem_ids to Problem objects
        """
        if difficulty_distribution is None:
            # Create a bell curve distribution favoring medium difficulty
            # with some easy and hard puzzles
            difficulty_distribution = {
                1: 0.05,  # Very easy
                2: 0.10, 
                3: 0.15,
                4: 0.20,  # Medium difficulty (most common)
                5: 0.20,
                6: 0.15,
                7: 0.10,
                8: 0.05   # Very difficult
            }
            
        # Normalize the distribution
        total = sum(difficulty_distribution.values())
        for k in difficulty_distribution:
            difficulty_distribution[k] /= total
            
        # Generate the specified number of puzzles
        puzzles = {}
        difficulties = []
        
        # Convert distribution to list of difficulties
        for difficulty, proportion in difficulty_distribution.items():
            count = int(proportion * num_puzzles)
            difficulties.extend([difficulty] * count)
            
        # Add remaining puzzles if there's a rounding issue
        while len(difficulties) < num_puzzles:
            difficulties.append(random.randint(1, 8))
            
        # Shuffle the difficulties
        random.shuffle(difficulties)
        
        for i in range(num_puzzles):
            difficulty = difficulties[i]
            puzzle = self.generate_forward_puzzle(difficulty)
            
            # Ensure unique problem_id
            while puzzle.problem_id in puzzles:
                puzzle.problem_id = f"{random.randint(1000, 9999)}"
                
            puzzles[puzzle.problem_id] = puzzle
            
            if (i + 1) % 10 == 0:
                logging.info(f"Generated {i + 1}/{num_puzzles} puzzles")
                
        return puzzles
        
    def save_puzzles(self, puzzles, output_dir):
        """Save generated puzzles to JSON files in the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        for problem_id, puzzle in puzzles.items():
            output_path = os.path.join(output_dir, f"{problem_id}.json")
            with open(output_path, 'w') as f:
                f.write(puzzle.model_dump_json(indent=4))
                
        logging.info(f"Saved {len(puzzles)} puzzles to {output_dir}")
        
if __name__ == "__main__":
    # Example usage
    generator = PuzzleGenerator()
    puzzles = generator.generate_puzzles(10)  # Generate 10 puzzles
    generator.save_puzzles(puzzles, "../dataset/puzzles")