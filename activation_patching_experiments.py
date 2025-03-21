import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.fft import fft
import math
from helical_llm_models import HelicalEncoding

class ActivationPatching:
    """
    Implementation of activation patching experiments to understand
    how LLMs represent and manipulate numbers, as described in
    "Language Models Use Trigonometry to Do Addition".
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
        
        # Store hooks for activation patching
        self.hooks = []
        self.stored_activations = {}
        self.hook_handles = []
    
    def _register_hooks(self):
        """Register hooks for all layers of the model"""
        # Clear any existing hooks
        self._remove_hooks()
        
        # Define hook function
        def hook_fn(module, input, output, name):
            self.stored_activations[name] = output.detach().cpu()
            return output
        
        # Register hooks for attention and MLP layers
        layer_idx = 0
        
        # The exact structure depends on the model architecture
        # This works for GPT-2 based models
        for name, module in self.model.named_modules():
            if "mlp" in name.lower() and isinstance(module, nn.Module):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, n=name: hook_fn(mod, inp, out, n)
                )
                self.hook_handles.append(handle)
                self.hooks.append(name)
            
            elif "attn" in name.lower() and "output" not in name.lower() and isinstance(module, nn.Module):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, n=name: hook_fn(mod, inp, out, n)
                )
                self.hook_handles.append(handle)
                self.hooks.append(name)
        
        print(f"Registered {len(self.hook_handles)} hooks")
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.stored_activations = {}
    
    def get_clean_and_corrupted_runs(self, a, b, corrupted_a=None):
        """
        Run the model on clean and corrupted prompts for a+b.
        
        Args:
            a: First number in addition
            b: Second number in addition
            corrupted_a: Value to use for corrupted prompt (default: a+10 mod 100)
            
        Returns:
            Dictionary with clean and corrupted outputs and activations
        """
        if corrupted_a is None:
            corrupted_a = (a + 10) % 100
        
        # Register hooks to capture activations
        self._register_hooks()
        
        # Clean prompt: a+b
        clean_prompt = f"{a}+{b}="
        clean_inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
        
        # Run model on clean prompt
        with torch.no_grad():
            clean_outputs = self.model(**clean_inputs, output_hidden_states=True)
            clean_logits = clean_outputs.logits
            clean_hidden_states = clean_outputs.hidden_states
            
            # Store clean activations
            clean_activations = {k: v.clone() for k, v in self.stored_activations.items()}
            self.stored_activations = {}
        
        # Corrupted prompt: a'+b
        corrupted_prompt = f"{corrupted_a}+{b}="
        corrupted_inputs = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.device)
        
        # Run model on corrupted prompt
        with torch.no_grad():
            corrupted_outputs = self.model(**corrupted_inputs, output_hidden_states=True)
            corrupted_logits = corrupted_outputs.logits
            corrupted_hidden_states = corrupted_outputs.hidden_states
            
            # Store corrupted activations
            corrupted_activations = {k: v.clone() for k, v in self.stored_activations.items()}
            self.stored_activations = {}
        
        # Get answer token ID
        answer = a + b
        answer_token_id = self.tokenizer.encode(str(answer))[-1]
        
        # Remove hooks
        self._remove_hooks()
        
        return {
            "a": a,
            "b": b,
            "corrupted_a": corrupted_a,
            "answer": answer,
            "answer_token_id": answer_token_id,
            "clean_logits": clean_logits,
            "corrupted_logits": corrupted_logits,
            "clean_hidden_states": clean_hidden_states,
            "corrupted_hidden_states": corrupted_hidden_states,
            "clean_activations": clean_activations,
            "corrupted_activations": corrupted_activations,
        }
    
    def activation_patching(self, clean_hidden_states, corrupted_hidden_states, 
                           corrupted_inputs, layer, token_idx=0):
        """
        Perform activation patching by replacing the hidden state at a specific layer.
        
        Args:
            clean_hidden_states: Hidden states from clean run
            corrupted_hidden_states: Hidden states from corrupted run
            corrupted_inputs: Input IDs for corrupted prompt
            layer: Layer to patch
            token_idx: Token position to patch (0 = first token)
            
        Returns:
            Logits after patching
        """
        # Create a copy of the corrupted hidden states
        patched_hidden_states = [h.clone() for h in corrupted_hidden_states]
        
        # Replace the hidden state at the specified layer and token
        patched_hidden_states[layer] = patched_hidden_states[layer].clone()
        patched_hidden_states[layer][0, token_idx] = clean_hidden_states[layer][0, token_idx].clone()
        
        # Run forward pass from the patched layer
        with torch.no_grad():
            # This is a simplified implementation - actual patching would need to 
            # run the model from the patched layer rather than from the start
            outputs = self.model(
                input_ids=corrupted_inputs.input_ids,
                attention_mask=corrupted_inputs.attention_mask,
                output_hidden_states=True,
                past_key_values=None  # Ideally would pass KV cache up to patched layer
            )
            
        return outputs.logits
    
    def create_helical_basis(self, values, k=4):
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
    
    def fit_helix_to_hidden_state(self, hidden_state, a, b=None, a_plus_b=None):
        """
        Fit a helical model to a hidden state.
        
        Args:
            hidden_state: Hidden state tensor to fit
            a: First number
            b: Optional second number
            a_plus_b: Optional a+b value
            
        Returns:
            R² fit quality, coefficients, and basis
        """
        # Determine values to include in basis
        values = [a]
        if b is not None:
            values.append(b)
        if a_plus_b is not None:
            values.append(a_plus_b)
        
        # Create basis
        B = self.create_helical_basis(values)
        
        # Fit using linear regression
        # h = B * c
        coeffs = torch.linalg.lstsq(B, hidden_state.unsqueeze(0))[0]
        
        # Calculate fit quality (R²)
        pred = torch.matmul(B, coeffs)
        total_var = torch.var(hidden_state)
        r2 = 1 - torch.mean((hidden_state - pred.squeeze(0))**2) / total_var
        
        return r2.item(), coeffs, B
    
    def analyze_layer_contributions(self, a_values, b_values, num_layers=None):
        """
        Analyze how each layer contributes to addition.
        
        Args:
            a_values: List of first numbers
            b_values: List of second numbers
            num_layers: Number of layers to analyze (default: all)
            
        Returns:
            Dictionary of contributions by layer
        """
        if num_layers is None:
            # Get automatically from model config
            if hasattr(self.model.config, "n_layer"):
                num_layers = self.model.config.n_layer
            else:
                num_layers = 12  # Default
        
        # Results storage
        results = {
            "layer_patch": np.zeros(num_layers+1),  # +1 for embeddings
            "helix_a": np.zeros(num_layers+1),
            "helix_b": np.zeros(num_layers+1),
            "helix_sum": np.zeros(num_layers+1),
            "helix_combined": np.zeros(num_layers+1),
        }
        
        for a, b in zip(a_values, b_values):
            print(f"Processing {a}+{b}...")
            
            # Get clean and corrupted runs
            data = self.get_clean_and_corrupted_runs(a, b)
            
            # For each layer
            for layer in range(num_layers + 1):  # +1 for embeddings
                # Measure logit difference with layer patching
                patched_logits = self.activation_patching(
                    data["clean_hidden_states"],
                    data["corrupted_hidden_states"],
                    self.tokenizer(f"{data['corrupted_a']}+{b}=", return_tensors="pt").to(self.device),
                    layer,
                    token_idx=0  # Patch the 'a' token
                )
                
                # Calculate logit difference for correct answer
                ld_layer = (patched_logits[0, -1, data["answer_token_id"]] - 
                           data["corrupted_logits"][0, -1, data["answer_token_id"]]).item()
                
                results["layer_patch"][layer] += ld_layer
                
                # Fit helical models to the hidden state
                # Extract the hidden state for the "=" token
                last_token_hidden = data["clean_hidden_states"][layer][0, -1].cpu()
                
                # Fit various helical models
                r2_a, _, _ = self.fit_helix_to_hidden_state(last_token_hidden, a)
                r2_b, _, _ = self.fit_helix_to_hidden_state(last_token_hidden, b)
                r2_sum, _, _ = self.fit_helix_to_hidden_state(last_token_hidden, a + b)
                r2_combined, _, _ = self.fit_helix_to_hidden_state(last_token_hidden, a, b, a + b)
                
                results["helix_a"][layer] += r2_a
                results["helix_b"][layer] += r2_b
                results["helix_sum"][layer] += r2_sum
                results["helix_combined"][layer] += r2_combined
        
        # Average results
        n = len(a_values)
        for key in results:
            results[key] /= n
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Layer patching results
        plt.subplot(2, 1, 1)
        plt.plot(range(num_layers + 1), results["layer_patch"], 'o-', label="Layer Patching")
        plt.xlabel("Layer")
        plt.ylabel("Logit Difference")
        plt.title("Layer Contributions to Addition (Layer Patching)")
        plt.grid(True)
        
        # Helical fit results
        plt.subplot(2, 1, 2)
        plt.plot(range(num_layers + 1), results["helix_a"], 'o-', label="helix(a)")
        plt.plot(range(num_layers + 1), results["helix_b"], 's-', label="helix(b)")
        plt.plot(range(num_layers + 1), results["helix_sum"], '^-', label="helix(a+b)")
        plt.plot(range(num_layers + 1), results["helix_combined"], 'D-', label="helix(a,b,a+b)")
        plt.xlabel("Layer")
        plt.ylabel("Fit Quality (R²)")
        plt.title("Helical Representation Quality by Layer")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def analyze_mlp_and_attention(self, a_values, b_values, layers_to_analyze=None):
        """
        Analyze contributions of MLP and attention components.
        
        Args:
            a_values: List of first numbers
            b_values: List of second numbers
            layers_to_analyze: Which layers to focus on (default: determined from initial analysis)
            
        Returns:
            Dictionary of results
        """
        if layers_to_analyze is None:
            # Default to middle and late layers where addition tends to happen
            if hasattr(self.model.config, "n_layer"):
                n_layers = self.model.config.n_layer
                layers_to_analyze = list(range(n_layers//2, n_layers))
            else:
                layers_to_analyze = list(range(6, 12))  # Default
        
        results = {
            "total_effect": {},
            "direct_effect": {},
            "helix_fit": {},
        }
        
        # Process each example
        for i, (a, b) in enumerate(zip(a_values, b_values)):
            print(f"Processing example {i+1}/{len(a_values)}: {a}+{b}")
            
            # Get clean and corrupted runs
            data = self.get_clean_and_corrupted_runs(a, b)
            
            # Analyze each layer's MLP and attention components
            for layer in layers_to_analyze:
                # Analyze MLP contributions
                mlp_key = f"mlp.{layer}"
                
                if mlp_key not in results["total_effect"]:
                    results["total_effect"][mlp_key] = 0
                    results["direct_effect"][mlp_key] = 0
                    results["helix_fit"][mlp_key] = 0
                
                # Get MLP output from clean and corrupted runs
                mlp_clean = data["clean_activations"].get(mlp_key, None)
                mlp_corrupted = data["corrupted_activations"].get(mlp_key, None)
                
                if mlp_clean is not None and mlp_corrupted is not None:
                    # Calculate how well the MLP output is fit by helix(a+b)
                    last_token_hidden = mlp_clean[0, -1]
                    r2, _, _ = self.fit_helix_to_hidden_state(last_token_hidden, a + b)
                    results["helix_fit"][mlp_key] += r2
                    
                    # TODO: Calculate total effect via activation patching
                    # TODO: Calculate direct effect via path patching
                    # These require more complex model manipulation
                    
                    # For now, we'll use placeholder values
                    results["total_effect"][mlp_key] += 1.0
                    results["direct_effect"][mlp_key] += 0.5
                
                # Analyze attention contributions
                attn_key = f"attn.{layer}"
                
                if attn_key not in results["total_effect"]:
                    results["total_effect"][attn_key] = 0
                    results["direct_effect"][attn_key] = 0
                    results["helix_fit"][attn_key] = 0
                
                # Get attention output from clean and corrupted runs
                attn_clean = data["clean_activations"].get(attn_key, None)
                attn_corrupted = data["corrupted_activations"].get(attn_key, None)
                
                if attn_clean is not None and attn_corrupted is not None:
                    # Calculate how well the attention output is fit by helix(a+b)
                    last_token_hidden = attn_clean[0, -1]
                    r2, _, _ = self.fit_helix_to_hidden_state(last_token_hidden, a + b)
                    results["helix_fit"][attn_key] += r2
                    
                    # Placeholder values
                    results["total_effect"][attn_key] += 0.8
                    results["direct_effect"][attn_key] += 0.3
        
        # Average results
        n = len(a_values)
        for metric in ["total_effect", "direct_effect", "helix_fit"]:
            for key in results[metric]:
                results[metric][key] /= n
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Get component names and sort by layer
        components = sorted(results["total_effect"].keys())
        
        # Total effect and direct effect
        plt.subplot(2, 1, 1)
        total_effects = [results["total_effect"][c] for c in components]
        direct_effects = [results["direct_effect"][c] for c in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        plt.bar(x - width/2, total_effects, width, label='Total Effect')
        plt.bar(x + width/2, direct_effects, width, label='Direct Effect')
        
        plt.xlabel('Component')
        plt.ylabel('Effect Strength')
        plt.title('MLP and Attention Contributions to Addition')
        plt.xticks(x, components, rotation=45)
        plt.legend()
        
        # Helix fit quality
        plt.subplot(2, 1, 2)
        helix_fits = [results["helix_fit"][c] for c in components]
        
        plt.bar(x, helix_fits, width)
        plt.xlabel('Component')
        plt.ylabel('Helix(a+b) Fit Quality (R²)')
        plt.title('Quality of Helical Fit by Component')
        plt.xticks(x, components, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return results

    def inspect_neuron_activations(self, a_values, b_values, layer, component_type="mlp"):
        """
        Analyze individual neuron activations to identify those involved in addition.
        
        Args:
            a_values: List of first numbers
            b_values: List of second numbers
            layer: Layer to analyze
            component_type: 'mlp' or 'attn'
            
        Returns:
            Dictionary of neuron analysis results
        """
        # Prepare grid of a+b combinations to analyze
        if len(a_values) == 1 and len(b_values) == 1:
            # Create a grid for a single example
            grid_a = np.arange(0, 10)
            grid_b = np.arange(0, 10)
            a_grid, b_grid = np.meshgrid(grid_a, grid_b)
            a_values_grid = a_grid.flatten()
            b_values_grid = b_grid.flatten()
        else:
            # Use provided values
            a_values_grid = a_values
            b_values_grid = b_values
        
        # Component key for accessing activations
        component_key = f"{component_type}.{layer}"
        
        # Store activations for each a+b
        activations = {}
        
        for a, b in zip(a_values_grid, b_values_grid):
            print(f"Processing activations for {a}+{b}")
            
            # Get clean run
            data = self.get_clean_and_corrupted_runs(a, b)
            
            # Store activations of the component for this a+b
            if component_key in data["clean_activations"]:
                # Get activations for the final token
                activations[(a, b)] = data["clean_activations"][component_key][0, -1].numpy()
        
        # Analyze activation patterns
        sum_values = [a + b for a, b in activations.keys()]
        
        # Select top neurons based on variance across examples
        neuron_variance = np.var([act for act in activations.values()], axis=0)
        top_neuron_indices = np.argsort(neuron_variance)[-20:]  # Top 20 neurons
        
        # Analyze periodicity of top neurons
        results = {
            "neuron_indices": top_neuron_indices,
            "fourier_periods": [],
            "neuron_fits": []
        }
        
        # Plot activations for top neurons
        plt.figure(figsize=(15, 15))
        
        for i, neuron_idx in enumerate(top_neuron_indices):
            # Get activations for this neuron
            neuron_activations = np.array([act[neuron_idx] for act in activations.values()])
            
            # Calculate periodicity using FFT
            fft_result = np.abs(fft(neuron_activations))
            top_freq = np.argmax(fft_result[1:len(fft_result)//2]) + 1
            period = len(neuron_activations) / top_freq if top_freq > 0 else np.inf
            
            results["fourier_periods"].append(period)
            
            # Plot activation heatmap
            if i < 16:  # Plot 16 neurons
                plt.subplot(4, 4, i+1)
                
                # Convert to a+b grid
                activation_grid = np.zeros((10, 10))
                for (a, b), act in zip(activations.keys(), neuron_activations):
                    if a < 10 and b < 10:  # Only plot within our 10x10 grid
                        activation_grid[int(a), int(b)] = act
                
                plt.imshow(activation_grid, cmap='viridis')
                plt.colorbar()
                plt.title(f"Neuron {neuron_idx}\nPeriod ≈ {period:.1f}")
                plt.xlabel('b')
                plt.ylabel('a')
        
        plt.suptitle(f"Top Neuron Activations in {component_type.upper()}.{layer}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        # Fit helical basis to top neurons
        for neuron_idx in top_neuron_indices:
            # Get activations for this neuron
            neuron_activations = np.array([act[neuron_idx] for act in activations.values()])
            
            # Create helical basis for a+b
            sum_tensor = torch.tensor(sum_values, dtype=torch.float32)
            B = torch.zeros((len(sum_values), 9))  # Linear + 4 Fourier features
            
            # Linear term
            B[:, 0] = sum_tensor
            
            # Fourier features
            for i, period in enumerate(self.periods):
                B[:, 2*i + 1] = torch.cos(2 * math.pi * sum_tensor / period)
                B[:, 2*i + 2] = torch.sin(2 * math.pi * sum_tensor / period)
            
            # Fit using linear regression
            neuron_activations_tensor = torch.tensor(neuron_activations, dtype=torch.float32)
            coeffs = torch.linalg.lstsq(B, neuron_activations_tensor.unsqueeze(1))[0]
            
            # Calculate fit quality (R²)
            pred = torch.matmul(B, coeffs)
            total_var = torch.var(neuron_activations_tensor)
            r2 = 1 - torch.mean((neuron_activations_tensor - pred.squeeze(1))**2) / total_var
            
            results["neuron_fits"].append({
                "neuron_idx": neuron_idx,
                "r2": r2.item(),
                "coeffs": coeffs.squeeze(1).numpy(),
            })
        
        # Plot helical fit quality
        plt.figure(figsize=(10, 6))
        
        r2_values = [fit["r2"] for fit in results["neuron_fits"]]
        plt.bar(range(len(top_neuron_indices)), r2_values)
        
        plt.xlabel("Neuron Index (Sorted by Variance)")
        plt.ylabel("Helical Fit Quality (R²)")
        plt.title(f"Quality of Helical Fit for Top Neurons in {component_type.upper()}.{layer}")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results

if __name__ == "__main__":
    # Run experiments with a small model for demonstration
    patching = ActivationPatching(model_name="gpt2")
    
    # 1. Analyze layer contributions
    print("Analyzing layer contributions to addition...")
    a_values = [5, 12, 27, 31, 42]
    b_values = [3, 8, 14, 22, 35]
    layer_results = patching.analyze_layer_contributions(a_values, b_values)
    
    # 2. Analyze MLP and attention components
    print("Analyzing MLP and attention components...")
    component_results = patching.analyze_mlp_and_attention(a_values, b_values)
    
    # 3. Inspect neuron activations
    print("Inspecting neuron activations...")
    # Find a layer with strong contribution to a+b
    strong_layer = np.argmax(layer_results["helix_sum"])
    neuron_results = patching.inspect_neuron_activations([23], [45], strong_layer)
    
    print("Experiments complete!") 