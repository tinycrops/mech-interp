# Helical Number Representation in LLMs

This repository contains the implementation of experiments and analyses related to the paper ["Language Models Use Trigonometry to Do Addition"](https://www.alignmentforum.org/posts/b4mthZHYcZzDuEgN8/language-models-use-trigonometry-to-do-addition) by Subhash Kantamneni and Max Tegmark.

## Overview

The project investigates how large language models (LLMs) represent and manipulate numbers internally. The key findings from the paper suggest that:

1. LLMs represent numbers as points on helices in their hidden states
2. Addition is performed using a "Clock algorithm" - a trigonometric mechanism where the model rotates one number by the value of the other to compute their sum

This implementation includes various experiments and visualizations to validate these claims and provide deeper insights into how LLMs perform numerical operations.

## Project Structure

- `helical_llm_models.py`: Implementation of helical encoding and a modified transformer with number-aware capabilities
- `helical_number_experiments.py`: Experiments to validate the helical representation hypothesis
- `activation_patching_experiments.py`: Activation patching to understand how different components contribute to addition
- `circuit_analysis.py`: Advanced causal tracing and circuit analysis to identify the addition mechanism
- `clock_algorithm_visualization.py`: Visualizations of the Clock algorithm for number addition
- `run_all_experiments.py`: Script to run all experiments together

## Key Experiments

### 1. Helical Representation Analysis
- Fourier analysis of number representations to identify periodic structure
- Visualization of number embeddings in helical subspaces
- Analysis of how helical representations evolve through model layers

### 2. Activation Patching Experiments
- Layer-wise contribution to addition tasks
- Analysis of MLP and attention components
- Identification of neurons involved in numerical operations

### 3. Circuit Analysis
- Causal tracing to identify the circuit responsible for addition
- Analysis of how the Clock algorithm is implemented in the model
- Correlation between helical representation quality and functional contribution

### 4. Clock Algorithm Visualizations
- Visual demonstration of number representation on helices
- Illustration of how the Clock algorithm performs addition
- Visualization of helical projections and relationships to trigonometric functions

## Usage

### Requirements

```
torch
numpy
matplotlib
transformers
scikit-learn
scipy
```

### Running Experiments

To run all experiments:

```bash
python run_all_experiments.py
```

To run specific experiments:

```bash
# Run just the helical representation analysis
python run_all_experiments.py --run_helical_subspace --run_fourier_analysis

# Run activation patching experiments
python run_all_experiments.py --run_layer_analysis --run_component_analysis

# Run with a specific model
python run_all_experiments.py --model_name gpt2-medium
```

### Visualization Options

```bash
# Run only visualizations
python run_all_experiments.py --run_helix_viz --run_clock_demo

# Save results without displaying plots (useful for headless environments)
python run_all_experiments.py --no_display
```

## Results and Findings

The experiments in this repository validate the key claims made in the paper:

1. **Helical Representations**: We confirm that LLMs represent numbers on helices with specific periods (primarily 10 and 100 for base-10 numbers).

2. **Clock Algorithm**: Our circuit analysis shows that LLMs implement a mechanism similar to the Clock algorithm for addition.

3. **Circuit Localization**: We identify the specific components (MLP layers and attention heads) responsible for the addition operation.

4. **Number-Specific Neurons**: We identify neurons that show periodic activation patterns aligned with specific digit places.

These findings provide significant insights into how neural networks can develop mathematical capabilities through training, even without explicit mathematical rules.

## Future Work

- Extend the analysis to other mathematical operations (subtraction, multiplication)
- Apply these findings to improve numerical reasoning in LLMs
- Explore the development of helical representations during model training
- Investigate how the helical structure changes with different number systems (binary, hexadecimal)

## Acknowledgements

This work is based on the research by Subhash Kantamneni and Max Tegmark, and builds upon their findings on helical representations in LLMs.

## License

MIT 