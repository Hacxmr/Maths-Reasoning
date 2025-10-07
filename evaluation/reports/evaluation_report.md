# SED Puzzle Evaluation Results

## Success Rates

| Model | Zero-shot | Few-shot | Chain of Thought |
|-------|-----------|----------|------------------|
| claude-3-sonnet | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) |
| gemini-1.5-flash | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) |
| gpt-4o | 34/100 (34.0%) | 45/100 (45.0%) | 61/100 (61.0%) |
| mock | 3/5 (60.0%) | 2/5 (40.0%) | 2/5 (40.0%) |
| openrouter-claude-3-sonnet | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) |
| openrouter-claude-3.5-sonnet | 1/2 (50.0%) | 0/2 (0.0%) | 0/2 (0.0%) |
| openrouter-gemini-1.5-pro | 0/3 (0.0%) | 0/3 (0.0%) | 0/0 (0.0%) |
| openrouter-gemini-2.5-pro | 2/2 (100.0%) | 2/2 (100.0%) | 0/2 (0.0%) |
| openrouter-gpt-4o | 1/2 (50.0%) | 1/2 (50.0%) | 1/2 (50.0%) |

## Analysis

### Key Findings

- The best performing configuration was **openrouter-gemini-2.5-pro** with **few-shot** prompting, achieving a 100.0% success rate.
- **Prompt Type Analysis**:
  - For claude-3-sonnet, cot prompting was most effective (0.0%).
  - For gemini-1.5-flash, cot prompting was most effective (0.0%).
  - For gpt-4o, cot prompting was most effective (61.0%).
  - For mock, zero-shot prompting was most effective (60.0%).
  - For openrouter-claude-3-sonnet, cot prompting was most effective (0.0%).
  - For openrouter-claude-3.5-sonnet, zero-shot prompting was most effective (50.0%).
  - For openrouter-gemini-1.5-pro, few-shot prompting was most effective (0.0%).
  - For openrouter-gemini-2.5-pro, few-shot prompting was most effective (100.0%).
  - For openrouter-gpt-4o, cot prompting was most effective (50.0%).
