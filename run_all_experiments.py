import os
import torch
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import all our experiment modules
from helical_number_experiments import HelicalExperiments
from activation_patching_experiments import ActivationPatching
from circuit_analysis import CircuitAnalysis
from clock_algorithm_visualization import ClockAlgorithmVisualization

def create_output_directory():
    """Create a directory for experiment outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiment_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir

def run_helical_experiments(args, output_dir):
    """Run experiments from the helical_number_experiments.py script"""
    print("\n" + "="*80)
    print("RUNNING HELICAL NUMBER EXPERIMENTS")
    print("="*80)
    
    experiments = HelicalExperiments(model_name=args.model_name)
    
    # Standard test sets for all experiments
    simple_examples = list(range(10, 50, 10))  # [10, 20, 30, 40]
    
    # 1. Analyze Fourier structure of number representations
    if args.run_fourier_analysis:
        print("\nAnalyzing Fourier structure of number representations...")
        plt.figure()
        experiments.analyze_fourier_structure(max_num=100)
        plt.savefig(f"{output_dir}/fourier_structure.png")
        if not args.no_display:
            plt.show()
    
    # 2. Visualize the helical subspace
    if args.run_helical_subspace:
        print("\nVisualizing helical subspace...")
        experiments.visualize_helix_subspace(list(range(0, 100, 10)))
        # This creates multiple figures, saved in the visualization function
        if not args.no_display:
            plt.show()
    
    # 3. Test addition mechanism
    if args.run_addition_mechanism:
        print("\nTesting addition mechanism...")
        # Create test set of addition problems
        a_values = [5, 12, 27, 31, 42]
        b_values = [3, 8, 14, 22, 35]
        results = experiments.analyze_addition_mechanism(a_values, b_values)
        
        # Save results
        plt.figure()
        plt.savefig(f"{output_dir}/addition_mechanism.png")
        if not args.no_display:
            plt.show()
    
    print("\nHelical experiments completed.")
    return experiments

def run_activation_patching(args, output_dir):
    """Run experiments from the activation_patching_experiments.py script"""
    print("\n" + "="*80)
    print("RUNNING ACTIVATION PATCHING EXPERIMENTS")
    print("="*80)
    
    patching = ActivationPatching(model_name=args.model_name)
    
    # Test data
    a_values = [5, 12, 27, 31, 42]
    b_values = [3, 8, 14, 22, 35]
    
    # 1. Analyze layer contributions
    if args.run_layer_analysis:
        print("\nAnalyzing layer contributions to addition...")
        layer_results = patching.analyze_layer_contributions(a_values, b_values)
        plt.savefig(f"{output_dir}/layer_contributions.png")
        if not args.no_display:
            plt.show()
    
    # 2. Analyze MLP and attention components
    if args.run_component_analysis:
        print("\nAnalyzing MLP and attention components...")
        component_results = patching.analyze_mlp_and_attention(a_values, b_values)
        plt.savefig(f"{output_dir}/component_analysis.png")
        if not args.no_display:
            plt.show()
    
    # 3. Inspect neuron activations
    if args.run_neuron_analysis:
        print("\nInspecting neuron activations...")
        # Find a layer with strong contribution to a+b
        strong_layer = 8  # Default value, ideally from layer_results
        if 'layer_results' in locals() and 'helix_sum' in layer_results:
            strong_layer = int(torch.argmax(torch.tensor(layer_results["helix_sum"])))
        
        neuron_results = patching.inspect_neuron_activations([23], [45], strong_layer)
        plt.savefig(f"{output_dir}/neuron_activations.png")
        if not args.no_display:
            plt.show()
    
    print("\nActivation patching experiments completed.")
    return patching

def run_circuit_analysis(args, output_dir):
    """Run experiments from the circuit_analysis.py script"""
    print("\n" + "="*80)
    print("RUNNING CIRCUIT ANALYSIS EXPERIMENTS")
    print("="*80)
    
    circuit = CircuitAnalysis(model_name=args.model_name)
    
    # Test data
    a_values = [5, 12, 27, 31, 42]
    b_values = [3, 8, 14, 22, 35]
    
    # 1. Analyze the addition circuit
    if args.run_circuit_analysis:
        print("\nAnalyzing the addition circuit...")
        circuit_results = circuit.analyze_addition_circuit(a_values, b_values)
        plt.savefig(f"{output_dir}/addition_circuit.png")
        if not args.no_display:
            plt.show()
    
    # 2. Validate the Clock algorithm
    if args.run_clock_validation:
        print("\nValidating the Clock algorithm...")
        clock_results = circuit.validate_clock_algorithm(a_values, b_values)
        plt.savefig(f"{output_dir}/clock_validation.png")
        if not args.no_display:
            plt.show()
    
    print("\nCircuit analysis experiments completed.")
    return circuit

def run_visualizations(args, output_dir):
    """Run visualizations from the clock_algorithm_visualization.py script"""
    print("\n" + "="*80)
    print("RUNNING CLOCK ALGORITHM VISUALIZATIONS")
    print("="*80)
    
    viz = ClockAlgorithmVisualization()
    
    # 1. Visualize number representation on a helix
    if args.run_helix_viz:
        print("\nVisualizing number representation on a helix...")
        viz.plot_number_helix(num_range=(0, 50), period=10)
        plt.savefig(f"{output_dir}/number_helix.png")
        if not args.no_display:
            plt.show()
    
    # 2. Demonstrate the Clock algorithm for addition
    if args.run_clock_demo:
        print("\nDemonstrating the Clock algorithm for addition...")
        
        # Example 1: Simple addition within period
        print("Example 1: Simple addition within period (5 + 3 = 8)")
        viz.visualize_addition_on_helix(5, 3, period=10)
        plt.savefig(f"{output_dir}/addition_5_3.png")
        if not args.no_display:
            plt.show()
        
        # Example 2: Addition with carry
        print("Example 2: Addition with carry (7 + 5 = 12)")
        viz.visualize_addition_on_helix(7, 5, period=10)
        plt.savefig(f"{output_dir}/addition_7_5.png")
        if not args.no_display:
            plt.show()
        
        # Example 3: Helix projection
        print("Example 3: Helical representation projected to a circle")
        viz.visualize_helix_projection(period=10)
        plt.savefig(f"{output_dir}/helix_projection.png")
        if not args.no_display:
            plt.show()
    
    print("\nClock algorithm visualizations completed.")
    return viz

def main():
    parser = argparse.ArgumentParser(description="Run helical number representation experiments")
    
    # Model selection
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name to use (default: gpt2)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for saving results (default: auto-generated)")
    parser.add_argument("--no_display", action="store_true",
                        help="Don't display plots (useful for headless environments)")
    
    # Experiment selection
    parser.add_argument("--run_all", action="store_true",
                        help="Run all experiments (default)")
    
    # Helical experiments
    parser.add_argument("--run_fourier_analysis", action="store_true",
                        help="Run Fourier analysis of number representations")
    parser.add_argument("--run_helical_subspace", action="store_true",
                        help="Run helical subspace visualization")
    parser.add_argument("--run_addition_mechanism", action="store_true",
                        help="Run addition mechanism analysis")
    
    # Activation patching experiments
    parser.add_argument("--run_layer_analysis", action="store_true",
                        help="Run layer contribution analysis")
    parser.add_argument("--run_component_analysis", action="store_true",
                        help="Run MLP and attention component analysis")
    parser.add_argument("--run_neuron_analysis", action="store_true",
                        help="Run neuron activation analysis")
    
    # Circuit analysis experiments
    parser.add_argument("--run_circuit_analysis", action="store_true",
                        help="Run addition circuit analysis")
    parser.add_argument("--run_clock_validation", action="store_true",
                        help="Run Clock algorithm validation")
    
    # Visualizations
    parser.add_argument("--run_helix_viz", action="store_true",
                        help="Run helix visualization")
    parser.add_argument("--run_clock_demo", action="store_true",
                        help="Run Clock algorithm demonstration")
    
    args = parser.parse_args()
    
    # If no specific experiments are selected, run all of them
    run_all = args.run_all or not any([
        args.run_fourier_analysis, args.run_helical_subspace, args.run_addition_mechanism,
        args.run_layer_analysis, args.run_component_analysis, args.run_neuron_analysis,
        args.run_circuit_analysis, args.run_clock_validation,
        args.run_helix_viz, args.run_clock_demo
    ])
    
    if run_all:
        args.run_fourier_analysis = True
        args.run_helical_subspace = True
        args.run_addition_mechanism = True
        args.run_layer_analysis = True
        args.run_component_analysis = True
        args.run_neuron_analysis = True
        args.run_circuit_analysis = True
        args.run_clock_validation = True
        args.run_helix_viz = True
        args.run_clock_demo = True
    
    # Create output directory if not specified
    output_dir = args.output_dir if args.output_dir else create_output_directory()
    
    # Print experiment configuration
    print(f"Running experiments with model: {args.model_name}")
    print(f"Results will be saved to: {output_dir}")
    
    # Run selected experiments
    # Each function returns the experiment object for potential reuse
    helical_exp = run_helical_experiments(args, output_dir)
    patching_exp = run_activation_patching(args, output_dir)
    circuit_exp = run_circuit_analysis(args, output_dir)
    viz_exp = run_visualizations(args, output_dir)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 