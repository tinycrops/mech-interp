import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import math
from helical_llm_models import HelicalEncoding

class CircuitAnalysis:
    """
    Implementation of circuit analysis techniques to identify and analyze
    circuits responsible for addition in LLMs, specifically focusing on
    the helical hypothesis from "Language Models Use Trigonometry to Do Addition".
    """
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize with a model and prepare environment.
        
        Args:
            model_name: HuggingFace model name to use for experiments
            device: Device to run experiments on
        """
        self.device = device
        self.model_name = model_name
        
        # Load model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded successfully")
        
        # Standard periods for helical basis
        self.periods = [2, 5, 10, 100]
        
        # Store hooks for circuit analysis
        self.hooks = []
        self.stored_activations = {}
        self.hook_handles = []
        
        # Define the set of layers to analyze
        if hasattr(self.model.config, "n_layer"):
            self.num_layers = self.model.config.n_layer
        else:
            self.num_layers = 12  # Default for GPT-2
    
    def _register_hooks(self):
        """Register hooks for all attention heads and MLP layers"""
        # Clear any existing hooks
        self._remove_hooks()
        
        # Define hook function
        def hook_fn(module, input, output, name):
            self.stored_activations[name] = output.detach().cpu()
            return output
        
        # Register hooks for attention blocks and MLP layers
        for layer_idx in range(self.num_layers):
            # Hook for each attention head
            try:
                attn_module = self.model.transformer.h[layer_idx].attn
                handle = attn_module.register_forward_hook(
                    lambda mod, inp, out, n=f"attn.{layer_idx}": hook_fn(mod, inp, out, n)
                )
                self.hook_handles.append(handle)
                self.hooks.append(f"attn.{layer_idx}")
                
                # Hook for MLP layer
                mlp_module = self.model.transformer.h[layer_idx].mlp
                handle = mlp_module.register_forward_hook(
                    lambda mod, inp, out, n=f"mlp.{layer_idx}": hook_fn(mod, inp, out, n)
                )
                self.hook_handles.append(handle)
                self.hooks.append(f"mlp.{layer_idx}")
            except (AttributeError, IndexError) as e:
                print(f"Error registering hooks for layer {layer_idx}: {e}")
                continue
        
        print(f"Registered {len(self.hook_handles)} hooks")
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.stored_activations = {}
    
    def causal_tracing(self, a_values, b_values, layers_to_analyze=None, heads_per_layer=12):
        """
        Perform causal tracing to identify the circuit responsible for addition.
        
        Args:
            a_values: List of first numbers in addition
            b_values: List of second numbers in addition
            layers_to_analyze: List of layer indices to analyze (default: all)
            heads_per_layer: Number of attention heads per layer
            
        Returns:
            Dictionary of causal effects for each component
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.num_layers))
        
        # Register hooks
        self._register_hooks()
        
        # Initialize results dictionary
        causal_effects = {
            "attn_heads": np.zeros((self.num_layers, heads_per_layer)),
            "mlp_layers": np.zeros(self.num_layers),
            "helical_fit": {
                "attn": np.zeros(self.num_layers),
                "mlp": np.zeros(self.num_layers)
            }
        }
        
        # Run causal tracing for each a+b example
        for a, b in zip(a_values, b_values):
            print(f"Performing causal tracing for {a}+{b}...")
            
            # Create prompt: a+b=
            clean_prompt = f"{a}+{b}="
            inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
            
            # Run the model to get clean activations
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                clean_logits = outputs.logits
                clean_hidden_states = outputs.hidden_states
                
                # Store clean activations
                clean_activations = {k: v.clone() for k, v in self.stored_activations.items()}
                self.stored_activations = {}
            
            # Get answer token ID
            answer = a + b
            answer_token_id = self.tokenizer.encode(str(answer))[-1]
            answer_logit = clean_logits[0, -1, answer_token_id].item()
            
            # Create a corrupted prompt: (a+10)+b=
            corrupted_a = (a + 10) % 100
            corrupted_prompt = f"{corrupted_a}+{b}="
            corrupted_inputs = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.device)
            
            # Run the model to get corrupted activations
            with torch.no_grad():
                corrupted_outputs = self.model(**corrupted_inputs, output_hidden_states=True)
                corrupted_logits = corrupted_outputs.logits
                
                # Store corrupted activations
                corrupted_activations = {k: v.clone() for k, v in self.stored_activations.items()}
                self.stored_activations = {}
            
            # Get corrupted logit for the correct answer
            corrupted_answer_logit = corrupted_logits[0, -1, answer_token_id].item()
            
            # For each layer and component, measure causal effect
            for layer_idx in layers_to_analyze:
                # MLP causal effect (by intervention)
                mlp_key = f"mlp.{layer_idx}"
                if mlp_key in clean_activations and mlp_key in corrupted_activations:
                    # Perform causal intervention by copying clean MLP activation to corrupted run
                    # This is a simplified representation - in practice, we would need to 
                    # rerun the forward pass with the patched activation
                    mlp_effect = self._estimate_causal_effect(
                        clean_activations[mlp_key],
                        corrupted_activations[mlp_key],
                        answer_logit,
                        corrupted_answer_logit
                    )
                    causal_effects["mlp_layers"][layer_idx] += mlp_effect
                    
                    # Calculate helical fit for the MLP output
                    r2 = self._calculate_helical_fit(clean_activations[mlp_key][0, -1], a + b)
                    causal_effects["helical_fit"]["mlp"][layer_idx] += r2
                
                # Attention layer causal effect
                attn_key = f"attn.{layer_idx}"
                if attn_key in clean_activations and attn_key in corrupted_activations:
                    # Estimate overall attention layer effect
                    attn_effect = self._estimate_causal_effect(
                        clean_activations[attn_key],
                        corrupted_activations[attn_key],
                        answer_logit,
                        corrupted_answer_logit
                    )
                    
                    # Estimate per-head effects (if output is split by heads)
                    # This is a simplified approximation
                    if clean_activations[attn_key].dim() >= 3:
                        head_dim = clean_activations[attn_key].size(-1) // heads_per_layer
                        for head_idx in range(min(heads_per_layer, clean_activations[attn_key].size(-1) // head_dim)):
                            start_idx = head_idx * head_dim
                            end_idx = (head_idx + 1) * head_dim
                            
                            head_effect = self._estimate_causal_effect(
                                clean_activations[attn_key][..., start_idx:end_idx],
                                corrupted_activations[attn_key][..., start_idx:end_idx],
                                answer_logit,
                                corrupted_answer_logit
                            )
                            causal_effects["attn_heads"][layer_idx, head_idx] += head_effect
                    
                    # Calculate helical fit for the attention output
                    r2 = self._calculate_helical_fit(clean_activations[attn_key][0, -1], a + b)
                    causal_effects["helical_fit"]["attn"][layer_idx] += r2
        
        # Average results across examples
        n_examples = len(a_values)
        causal_effects["mlp_layers"] /= n_examples
        causal_effects["attn_heads"] /= n_examples
        causal_effects["helical_fit"]["attn"] /= n_examples
        causal_effects["helical_fit"]["mlp"] /= n_examples
        
        # Remove hooks
        self._remove_hooks()
        
        # Visualize results
        self._visualize_causal_effects(causal_effects)
        
        return causal_effects
    
    def _estimate_causal_effect(self, clean_activation, corrupted_activation, clean_logit, corrupted_logit):
        """
        Estimate the causal effect of a component on the addition task.
        
        Args:
            clean_activation: Activation from clean run
            corrupted_activation: Activation from corrupted run
            clean_logit: Logit for correct answer in clean run
            corrupted_logit: Logit for correct answer in corrupted run
            
        Returns:
            Estimated causal effect (0 to 1 scale)
        """
        # This is a simplified approximation of causal effect
        # In a complete implementation, we would run the model forward from this point
        # with the patched activations
        
        # Measure norm difference between activations
        activation_diff = torch.norm(clean_activation - corrupted_activation).item()
        
        # Normalize by activation size
        activation_size = torch.norm(clean_activation).item()
        if activation_size > 0:
            activation_diff /= activation_size
        
        # Logit difference
        logit_diff = abs(clean_logit - corrupted_logit)
        
        # Combine the two measures (this is simplified)
        # Effect ranges from 0 to 1
        effect = min(1.0, activation_diff * logit_diff / 10)
        
        return effect
    
    def _calculate_helical_fit(self, hidden_state, value):
        """
        Calculate how well a hidden state fits a helical representation.
        
        Args:
            hidden_state: Hidden state tensor
            value: Value to create helical basis for
            
        Returns:
            R² of the fit
        """
        # Create helical basis
        basis = self._create_helical_basis([value])
        
        # Fit linear regression
        coeffs = torch.linalg.lstsq(basis, hidden_state.unsqueeze(0))[0]
        
        # Calculate fit quality (R²)
        pred = torch.matmul(basis, coeffs)
        total_var = torch.var(hidden_state)
        if total_var == 0:
            return 0.0
            
        r2 = 1 - torch.mean((hidden_state - pred.squeeze(0))**2) / total_var
        
        return r2.item()
    
    def _create_helical_basis(self, values, k=4):
        """
        Create a helical basis matrix for the given values.
        
        Args:
            values: List of numbers to create basis for
            k: Number of Fourier features
            
        Returns:
            Basis matrix of shape [len(values), 2*k+1]
        """
        values_tensor = torch.tensor(values, dtype=torch.float32)
        B = torch.zeros((len(values), 2*k + 1))
        
        # Linear term
        B[:, 0] = values_tensor
        
        # Fourier features
        for i, period in enumerate(self.periods[:k]):
            B[:, 2*i + 1] = torch.cos(2 * math.pi * values_tensor / period)
            B[:, 2*i + 2] = torch.sin(2 * math.pi * values_tensor / period)
        
        return B
    
    def _visualize_causal_effects(self, causal_effects):
        """
        Visualize the causal effects of different components.
        
        Args:
            causal_effects: Dictionary of causal effects from causal_tracing
        """
        # Plot MLP and attention layer effects
        plt.figure(figsize=(15, 12))
        
        # MLP layer effects
        plt.subplot(2, 2, 1)
        plt.bar(range(len(causal_effects["mlp_layers"])), causal_effects["mlp_layers"])
        plt.xlabel("Layer")
        plt.ylabel("Causal Effect")
        plt.title("MLP Layer Effects on Addition")
        plt.grid(True, alpha=0.3)
        
        # Attention head effects as heatmap
        plt.subplot(2, 2, 2)
        plt.imshow(causal_effects["attn_heads"], aspect='auto', interpolation='none', cmap='viridis')
        plt.colorbar(label="Causal Effect")
        plt.xlabel("Attention Head")
        plt.ylabel("Layer")
        plt.title("Attention Head Effects on Addition")
        
        # Helical fit for MLP and attention
        plt.subplot(2, 2, 3)
        plt.plot(range(len(causal_effects["helical_fit"]["mlp"])), 
                causal_effects["helical_fit"]["mlp"], 'o-', label="MLP Layers")
        plt.plot(range(len(causal_effects["helical_fit"]["attn"])), 
                causal_effects["helical_fit"]["attn"], 's-', label="Attention Layers")
        plt.xlabel("Layer")
        plt.ylabel("Helical Fit Quality (R²)")
        plt.title("Helical Fit Quality by Layer and Component")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined view: causal effect vs. helical fit
        plt.subplot(2, 2, 4)
        plt.scatter(causal_effects["mlp_layers"], causal_effects["helical_fit"]["mlp"], 
                   label="MLP Layers", marker='o', s=100)
        
        # Calculate attention layer effects (average across heads)
        attn_layer_effects = np.mean(causal_effects["attn_heads"], axis=1)
        plt.scatter(attn_layer_effects, causal_effects["helical_fit"]["attn"], 
                   label="Attention Layers", marker='s', s=100)
        
        # Add layer indices as text
        for i in range(len(causal_effects["mlp_layers"])):
            plt.text(causal_effects["mlp_layers"][i], causal_effects["helical_fit"]["mlp"][i], 
                    f"MLP {i}", fontsize=8)
            plt.text(attn_layer_effects[i], causal_effects["helical_fit"]["attn"][i], 
                    f"Attn {i}", fontsize=8)
        
        plt.xlabel("Causal Effect")
        plt.ylabel("Helical Fit Quality (R²)")
        plt.title("Relationship Between Causal Effect and Helical Representation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_addition_circuit(self, a_values, b_values):
        """
        Analyze the circuit responsible for addition.
        
        Args:
            a_values: List of first numbers in addition
            b_values: List of second numbers in addition
            
        Returns:
            Dictionary of circuit analysis results
        """
        # 1. Perform causal tracing to identify important components
        causal_effects = self.causal_tracing(a_values, b_values)
        
        # 2. Identify the top-3 most important components
        mlp_importance = [(i, effect) for i, effect in enumerate(causal_effects["mlp_layers"])]
        attn_importance = []
        for layer in range(causal_effects["attn_heads"].shape[0]):
            for head in range(causal_effects["attn_heads"].shape[1]):
                attn_importance.append((layer, head, causal_effects["attn_heads"][layer, head]))
        
        # Sort by effect size
        mlp_importance.sort(key=lambda x: x[1], reverse=True)
        attn_importance.sort(key=lambda x: x[2], reverse=True)
        
        # Get top components
        top_mlp = mlp_importance[:3]
        top_attn = attn_importance[:3]
        
        print("\nTop MLP layers for addition:")
        for idx, effect in top_mlp:
            print(f"  MLP layer {idx}: effect = {effect:.4f}")
        
        print("\nTop attention heads for addition:")
        for layer, head, effect in top_attn:
            print(f"  Layer {layer}, Head {head}: effect = {effect:.4f}")
        
        # 3. Perform more detailed analysis on top components
        # For this simplified version, we're just identifying them
        
        return {
            "causal_effects": causal_effects,
            "top_mlp": top_mlp,
            "top_attn": top_attn
        }

    def validate_clock_algorithm(self, a_values, b_values, layer_indices=None):
        """
        Validate the "Clock algorithm" hypothesis by analyzing how
        the model manipulates helical representations during addition.
        
        Args:
            a_values: List of first numbers in addition
            b_values: List of second numbers in addition
            layer_indices: Optional specific layers to analyze
            
        Returns:
            Dictionary of validation results
        """
        # If no specific layers provided, analyze all layers
        if layer_indices is None:
            if hasattr(self.model.config, "n_layer"):
                layer_indices = list(range(self.model.config.n_layer))
            else:
                layer_indices = list(range(12))  # Default for GPT-2
        
        # Register hooks
        self._register_hooks()
        
        # Initialize results
        results = {
            "layer_indices": layer_indices,
            "correlation_a": np.zeros(len(layer_indices)),
            "correlation_b": np.zeros(len(layer_indices)),
            "correlation_sum": np.zeros(len(layer_indices)),
            "clock_evidence": np.zeros(len(layer_indices)),
            "helix_params": [[] for _ in range(len(layer_indices))]
        }
        
        # For each a+b pair
        for a, b in zip(a_values, b_values):
            print(f"Validating Clock algorithm for {a}+{b}...")
            
            # Create prompt and run model
            prompt = f"{a}+{b}="
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Collect activations at each layer
                clean_activations = {k: v.clone() for k, v in self.stored_activations.items()}
                self.stored_activations = {}
            
            # For each layer
            for i, layer_idx in enumerate(layer_indices):
                # Extract relevant activations
                attn_key = f"attn.{layer_idx}"
                mlp_key = f"mlp.{layer_idx}"
                hidden_state = outputs.hidden_states[layer_idx+1][0, -1].cpu()  # +1 because [0] is embeddings
                
                # Test correlation with each value: a, b, a+b
                # We'll use helical projections for this
                corr_a = self._test_helical_correlation(hidden_state, a)
                corr_b = self._test_helical_correlation(hidden_state, b)
                corr_sum = self._test_helical_correlation(hidden_state, a + b)
                
                results["correlation_a"][i] += corr_a
                results["correlation_b"][i] += corr_b
                results["correlation_sum"][i] += corr_sum
                
                # Test if the Clock algorithm is being used
                # If so, we should see evidence of rotation by 'b' in the helical space
                clock_evidence = self._test_clock_algorithm(hidden_state, a, b)
                results["clock_evidence"][i] += clock_evidence
                
                # Store helical parameters for this layer
                if attn_key in clean_activations:
                    # Fit helical model to attention output for the last token
                    params = self._extract_helical_parameters(clean_activations[attn_key][0, -1], a + b)
                    results["helix_params"][i].append(params)
        
        # Average results across examples
        n_examples = len(a_values)
        results["correlation_a"] /= n_examples
        results["correlation_b"] /= n_examples
        results["correlation_sum"] /= n_examples
        results["clock_evidence"] /= n_examples
        
        # Remove hooks
        self._remove_hooks()
        
        # Visualize results
        self._visualize_clock_validation(results)
        
        return results
    
    def _test_helical_correlation(self, hidden_state, value):
        """
        Test how well a hidden state correlates with a helical representation.
        
        Args:
            hidden_state: Hidden state tensor
            value: Value to create helical basis for
            
        Returns:
            Correlation coefficient
        """
        # Create helical basis for the value
        basis = self._create_helical_basis([value])
        
        # Project hidden state onto basis
        coeffs = torch.linalg.lstsq(basis, hidden_state.unsqueeze(0))[0]
        projection = torch.matmul(basis, coeffs).squeeze(0)
        
        # Calculate correlation
        pearson_corr = pearsonr(hidden_state.numpy(), projection.numpy())[0]
        
        return pearson_corr
    
    def _test_clock_algorithm(self, hidden_state, a, b):
        """
        Test if the Clock algorithm is being used for addition.
        
        Args:
            hidden_state: Hidden state tensor
            a: First number
            b: Second number
            
        Returns:
            Evidence score for the Clock algorithm (0 to 1)
        """
        # Extract the helical components for period 10 (assuming base-10)
        # This is a simplification - a full implementation would analyze multiple periods
        period = 10
        
        # Create PCA to extract 2D plane where the helix might lie
        pca = PCA(n_components=2)
        hidden_2d = pca.fit_transform(hidden_state.unsqueeze(0).numpy())[0]
        
        # Calculate expected angular positions for a, b, and a+b
        angle_a = 2 * np.pi * a / period
        angle_b = 2 * np.pi * b / period
        angle_sum = 2 * np.pi * (a + b) / period
        angle_sum_mod = angle_sum % (2 * np.pi)
        
        # If the Clock algorithm is used, we'd expect to see:
        # 1. Hidden state angle corresponds to angle_sum_mod
        hidden_angle = np.arctan2(hidden_2d[1], hidden_2d[0])
        if hidden_angle < 0:
            hidden_angle += 2 * np.pi
            
        # Calculate angle difference (normalized to [0, 1])
        angle_diff = min(abs(hidden_angle - angle_sum_mod), 2*np.pi - abs(hidden_angle - angle_sum_mod)) / np.pi
        
        # Transform to a 0-1 score where 1 means perfect match
        evidence = 1 - angle_diff
        
        return evidence
    
    def _extract_helical_parameters(self, hidden_state, value):
        """
        Extract parameters of the helical representation from a hidden state.
        
        Args:
            hidden_state: Hidden state tensor
            value: Value being represented
            
        Returns:
            Dictionary of helical parameters
        """
        # Create helical basis for the value
        basis = self._create_helical_basis([value])
        
        # Fit the basis to the hidden state
        coeffs = torch.linalg.lstsq(basis, hidden_state.unsqueeze(0))[0]
        
        # Extract parameters for each period
        params = {}
        for i, period in enumerate(self.periods[:4]):
            # Calculate amplitude from cos and sin coefficients
            cos_idx = 2*i + 1
            sin_idx = 2*i + 2
            
            if cos_idx < coeffs.shape[0] and sin_idx < coeffs.shape[0]:
                cos_coeff = coeffs[cos_idx].item()
                sin_coeff = coeffs[sin_idx].item()
                
                amplitude = np.sqrt(cos_coeff**2 + sin_coeff**2)
                phase = np.arctan2(sin_coeff, cos_coeff)
                
                params[f"period_{period}"] = {
                    "amplitude": amplitude,
                    "phase": phase
                }
        
        return params
    
    def _visualize_clock_validation(self, results):
        """
        Visualize the results of Clock algorithm validation.
        
        Args:
            results: Results dictionary from validate_clock_algorithm
        """
        plt.figure(figsize=(15, 10))
        
        # Plot correlations
        plt.subplot(2, 2, 1)
        plt.plot(results["layer_indices"], results["correlation_a"], 'o-', label="Correlation with a")
        plt.plot(results["layer_indices"], results["correlation_b"], 's-', label="Correlation with b")
        plt.plot(results["layer_indices"], results["correlation_sum"], '^-', label="Correlation with a+b")
        plt.xlabel("Layer")
        plt.ylabel("Correlation")
        plt.title("Correlation with Helical Representations")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Clock algorithm evidence
        plt.subplot(2, 2, 2)
        plt.bar(results["layer_indices"], results["clock_evidence"])
        plt.xlabel("Layer")
        plt.ylabel("Evidence Score")
        plt.title("Evidence for the Clock Algorithm")
        plt.grid(True, alpha=0.3)
        
        # Plot evolution of correlations
        plt.subplot(2, 2, 3)
        plt.plot(results["correlation_a"], label="a")
        plt.plot(results["correlation_b"], label="b")
        plt.plot(results["correlation_sum"], label="a+b")
        plt.xlabel("Layer Progression")
        plt.ylabel("Correlation Strength")
        plt.title("Evolution of Representations Through Layers")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot relationship between sum correlation and Clock evidence
        plt.subplot(2, 2, 4)
        plt.scatter(results["correlation_sum"], results["clock_evidence"])
        
        # Add layer indices as text
        for i, layer_idx in enumerate(results["layer_indices"]):
            plt.text(results["correlation_sum"][i], results["clock_evidence"][i], 
                    f"{layer_idx}", fontsize=8)
        
        plt.xlabel("Correlation with a+b")
        plt.ylabel("Clock Algorithm Evidence")
        plt.title("Relationship Between Sum Representation and Clock Algorithm")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Run circuit analysis with a small model for demonstration
    circuit = CircuitAnalysis(model_name="gpt2")
    
    # Example addition problems
    a_values = [5, 12, 27, 31, 42]
    b_values = [3, 8, 14, 22, 35]
    
    # 1. Analyze the addition circuit
    print("Analyzing the addition circuit...")
    circuit_results = circuit.analyze_addition_circuit(a_values, b_values)
    
    # 2. Validate the Clock algorithm
    print("\nValidating the Clock algorithm...")
    clock_results = circuit.validate_clock_algorithm(a_values, b_values)
    
    print("\nCircuit analysis complete!") 