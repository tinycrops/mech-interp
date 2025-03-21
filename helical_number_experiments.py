import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.fft import fft
import math
from helical_llm_models import HelicalEncoding

class HelicalExperiments:
    """
    Implementation of experiments to validate the findings from 
    "Language Models Use Trigonometry to Do Addition" paper.
    """
    def __init__(self, model_name="gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize with a model and prepare environment.
        
        Args:
            model_name: HuggingFace model name to use for experiments
            device: Device to run experiments on
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # We'll load the model on-demand to save memory
        
        # Periods to use for helical bases
        self.periods = [2, 5, 10, 100]
        
        print(f"Initialized HelicalExperiments with {model_name} on {device}")
    
    def load_model(self):
        """Load model and tokenizer from HuggingFace"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully")
        except ImportError:
            print("Transformers library not found. Please install with: pip install transformers")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        
        return True

    def get_number_representations(self, numbers, layer=0):
        """
        Get the model's representation of numbers at a specific layer.
        
        Args:
            numbers: List of integers to represent
            layer: Layer to extract representations from
            
        Returns:
            Tensor of shape [len(numbers), hidden_dim]
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        representations = []
        
        with torch.no_grad():
            for num in numbers:
                # Format number with space prefix as in the paper
                text = f" {num}"
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                # Run model and get hidden states
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get the hidden state for the number token at specified layer
                # -1 index because we want the last token (the number token)
                hidden_state = outputs.hidden_states[layer][0, -1].cpu()
                
                representations.append(hidden_state)
        
        return torch.stack(representations)

    def analyze_fourier_structure(self, max_num=360):
        """
        Analyze the Fourier structure of number representations.
        Replicates Figure 2 from the paper.
        
        Args:
            max_num: Maximum number to analyze
            
        Returns:
            Plot of Fourier coefficients
        """
        numbers = list(range(max_num + 1))
        representations = self.get_number_representations(numbers)
        
        if representations is None:
            return
        
        # Center the representations
        representations = representations - representations.mean(dim=0, keepdim=True)
        
        # Compute Fourier transform along the number dimension
        # We need to transpose to get [hidden_dim, num_numbers]
        representations_t = representations.transpose(0, 1)
        
        # Compute FFT for each dimension
        fft_magnitudes = []
        for dim in range(representations_t.shape[0]):
            fft_result = fft(representations_t[dim].numpy())
            magnitude = np.abs(fft_result)
            fft_magnitudes.append(magnitude)
        
        # Average across hidden dimensions
        avg_magnitude = np.mean(fft_magnitudes, axis=0)
        
        # Plot the result
        plt.figure(figsize=(10, 6))
        # Only plot the first half (the rest is redundant due to symmetry)
        plt.plot(np.arange(len(avg_magnitude)//2), avg_magnitude[:len(avg_magnitude)//2])
        plt.title(f"Fourier Structure of Number Representations (Layer {0})")
        plt.xlabel("Frequency")
        plt.ylabel("Average Magnitude")
        
        # Add markers for key periods
        for period in [2, 5, 10, 100]:
            freq = max_num / period
            plt.axvline(x=freq, color='r', linestyle='--', alpha=0.5)
            plt.text(freq, plt.ylim()[1]*0.9, f"T={period}", rotation=90)
        
        plt.tight_layout()
        plt.show()
        
        return avg_magnitude

    def fit_helix(self, numbers, representations=None, k=4):
        """
        Fit a helical model to number representations.
        
        Args:
            numbers: List of numbers to fit
            representations: Pre-computed representations or None to compute them
            k: Number of Fourier features to use
            
        Returns:
            C: Coefficient matrix mapping from basis to full space
            B: Basis matrix for the helical representation
        """
        if representations is None:
            representations = self.get_number_representations(numbers)
            
        if representations is None:
            return None, None
        
        # Convert numbers to tensor
        numbers_tensor = torch.tensor(numbers, dtype=torch.float32)
        
        # Create PCA for dimensionality reduction
        pca = PCA(n_components=100)
        representations_reduced = pca.fit_transform(representations.numpy())
        representations_reduced = torch.tensor(representations_reduced, dtype=torch.float32)
        
        # Define the basis functions B(a)
        # [a, cos(2π/T₁*a), sin(2π/T₁*a), ..., cos(2π/Tₖ*a), sin(2π/Tₖ*a)]
        B = torch.zeros((len(numbers), 2*k + 1))
        
        # Linear term
        B[:, 0] = numbers_tensor
        
        # Fourier features
        for i, period in enumerate(self.periods[:k]):
            B[:, 2*i + 1] = torch.cos(2 * math.pi * numbers_tensor / period)
            B[:, 2*i + 2] = torch.sin(2 * math.pi * numbers_tensor / period)
        
        # Fit using linear regression (minimizing ||PCA(h) - C_PCA · B(a)^T||)
        C_PCA = torch.linalg.lstsq(B, representations_reduced)[0].T
        
        # Transform back to full dimensionality
        pca_components = torch.tensor(pca.components_, dtype=torch.float32)
        C = torch.matmul(pca_components.T, C_PCA)
        
        return C, B

    def validate_helix_fit(self, a_values, b_values, layer=0):
        """
        Validate the helix model by testing on addition problems.
        
        Args:
            a_values: List of 'a' values for a+b problems
            b_values: List of 'b' values for a+b problems
            layer: Layer to extract representations from
            
        Returns:
            Logit differences for different models
        """
        # Get representations for a
        a_representations = self.get_number_representations(a_values, layer)
        
        # Fit helix to a
        C_a, B_a = self.fit_helix(a_values, a_representations)
        
        # Create baseline methods
        results = {
            "layer_patch": [],
            "pca": [],
            "helix": [],
            "circle": [],
            "polynomial": [],
        }
        
        # For each a+b problem
        for i, (a, b) in enumerate(zip(a_values, b_values)):
            # Clean prompt: a+b
            clean_prompt = f"{a}+{b}="
            clean_inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
            
            # Corrupt prompt: a'+b where a' ≠ a
            corrupted_a = (a + 10) % 100  # Simple corruption
            corrupted_prompt = f"{corrupted_a}+{b}="
            corrupted_inputs = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.device)
            
            # Get clean and corrupted outputs
            with torch.no_grad():
                clean_outputs = self.model(**clean_inputs, output_hidden_states=True)
                corrupted_outputs = self.model(**corrupted_inputs, output_hidden_states=True)
                
                # Get the hidden state for a at the specified layer
                clean_a_hidden = clean_outputs.hidden_states[layer][0, 0].cpu()
                
                # Get logits for clean and corrupted
                clean_logits = clean_outputs.logits[0, -1]
                corrupted_logits = corrupted_outputs.logits[0, -1]
                
                # Correct answer token ID
                answer = a + b
                answer_token_id = self.tokenizer.encode(str(answer))[-1]
                
                # Original logit difference
                original_logit_diff = (corrupted_logits[answer_token_id] - clean_logits[answer_token_id]).item()
                
                # Layer patching
                layer_patched_outputs = self.model(**corrupted_inputs, output_hidden_states=True)
                
                # TODO: Implement proper activation patching here by replacing
                # hidden states at layer for the 'a' token with clean hidden states
                
                # TODO: Implement PCA, helix, circle, and polynomial models
                
                # For now, we'll add placeholder results
                results["layer_patch"].append(original_logit_diff)
                results["pca"].append(original_logit_diff * 0.8)  # Placeholder
                results["helix"].append(original_logit_diff * 0.9)  # Placeholder
                results["circle"].append(original_logit_diff * 0.85)  # Placeholder
                results["polynomial"].append(original_logit_diff * 0.4)  # Placeholder
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(a_values)} examples")
                
        return results

    def visualize_helix_subspace(self, numbers):
        """
        Visualize the helical subspace by projecting numbers onto it.
        Replicates Figure 3 from the paper.
        
        Args:
            numbers: List of numbers to visualize
            
        Returns:
            Plot of numbers in helical subspace
        """
        representations = self.get_number_representations(numbers)
        
        if representations is None:
            return
        
        # Fit helix to representations
        C, B = self.fit_helix(numbers, representations)
        
        # Project representations onto helical subspace
        # C† · h = B(a)
        C_pinv = torch.pinverse(C.T)
        projections = torch.matmul(representations, C_pinv)
        
        # Visualize each Fourier feature
        plt.figure(figsize=(15, 12))
        
        # For each Fourier feature (excluding linear component)
        for i, period in enumerate(self.periods[:4]):
            plt.subplot(2, 2, i+1)
            
            # Get cos and sin components for this period
            cos_idx = 2*i + 1
            sin_idx = 2*i + 2
            
            cos_values = projections[:, cos_idx].numpy()
            sin_values = projections[:, sin_idx].numpy()
            
            # Color by number value
            scatter = plt.scatter(cos_values, sin_values, c=numbers, cmap='viridis')
            
            # Plot unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
            
            # Group points by modulo
            for mod_val in range(period):
                mod_indices = [j for j, num in enumerate(numbers) if num % period == mod_val]
                if mod_indices:
                    mean_cos = np.mean(cos_values[mod_indices])
                    mean_sin = np.mean(sin_values[mod_indices])
                    plt.scatter(mean_cos, mean_sin, s=100, color='red', edgecolor='black')
                    plt.text(mean_cos*1.1, mean_sin*1.1, f"{mod_val} mod {period}")
            
            plt.title(f"Period = {period}")
            plt.xlabel("cos(2πa/T)")
            plt.ylabel("sin(2πa/T)")
            plt.axis('equal')
            
        # Also visualize the linear component
        plt.figure(figsize=(8, 6))
        plt.plot(numbers, projections[:, 0].numpy(), 'o-')
        plt.title("Linear Component")
        plt.xlabel("Number Value (a)")
        plt.ylabel("Projection Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_addition_mechanism(self, a_values, b_values, layer_range=(14, 28)):
        """
        Analyze how the model combines a and b to compute a+b.
        Tests the "Clock algorithm" hypothesis.
        
        Args:
            a_values: List of 'a' values
            b_values: List of 'b' values
            layer_range: Range of layers to analyze
            
        Returns:
            Analysis of how a+b is computed across layers
        """
        results = {}
        
        for layer in range(layer_range[0], layer_range[1]+1):
            print(f"Analyzing layer {layer}...")
            
            # Process each a+b problem
            helix_a_fits = []
            helix_b_fits = []
            helix_sum_fits = []
            helix_combined_fits = []
            
            for a, b in zip(a_values, b_values):
                # Create prompt
                prompt = f"{a}+{b}="
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    # Get hidden state for the "=" token, which should predict a+b
                    last_token_hidden = outputs.hidden_states[layer][0, -1].cpu()
                    
                    # Create helical basis for a, b, and a+b
                    sum_val = a + b
                    B_a = self.create_helical_basis([a])
                    B_b = self.create_helical_basis([b])
                    B_sum = self.create_helical_basis([sum_val])
                    B_combined = self.create_helical_basis([a, b, sum_val])
                    
                    # Fit each basis to the hidden state using linear regression
                    fit_a = torch.linalg.lstsq(B_a, last_token_hidden.unsqueeze(0))[0]
                    fit_b = torch.linalg.lstsq(B_b, last_token_hidden.unsqueeze(0))[0]
                    fit_sum = torch.linalg.lstsq(B_sum, last_token_hidden.unsqueeze(0))[0]
                    fit_combined = torch.linalg.lstsq(B_combined, last_token_hidden.unsqueeze(0))[0]
                    
                    # Calculate fit quality (R²)
                    pred_a = torch.matmul(B_a, fit_a)
                    pred_b = torch.matmul(B_b, fit_b)
                    pred_sum = torch.matmul(B_sum, fit_sum)
                    pred_combined = torch.matmul(B_combined, fit_combined)
                    
                    # Calculate R² for each fit
                    total_var = torch.var(last_token_hidden)
                    r2_a = 1 - torch.mean((last_token_hidden - pred_a.squeeze(0))**2) / total_var
                    r2_b = 1 - torch.mean((last_token_hidden - pred_b.squeeze(0))**2) / total_var
                    r2_sum = 1 - torch.mean((last_token_hidden - pred_sum.squeeze(0))**2) / total_var
                    r2_combined = 1 - torch.mean((last_token_hidden - pred_combined.squeeze(0))**2) / total_var
                    
                    helix_a_fits.append(r2_a.item())
                    helix_b_fits.append(r2_b.item())
                    helix_sum_fits.append(r2_sum.item())
                    helix_combined_fits.append(r2_combined.item())
            
            # Average results for this layer
            results[layer] = {
                "helix(a)": np.mean(helix_a_fits),
                "helix(b)": np.mean(helix_b_fits),
                "helix(a+b)": np.mean(helix_sum_fits),
                "helix(a,b,a+b)": np.mean(helix_combined_fits)
            }
        
        # Plot results
        layers = list(results.keys())
        
        plt.figure(figsize=(12, 6))
        plt.plot(layers, [results[l]["helix(a)"] for l in layers], 'o-', label="helix(a)")
        plt.plot(layers, [results[l]["helix(b)"] for l in layers], 's-', label="helix(b)")
        plt.plot(layers, [results[l]["helix(a+b)"] for l in layers], '^-', label="helix(a+b)")
        plt.plot(layers, [results[l]["helix(a,b,a+b)"] for l in layers], 'D-', label="helix(a,b,a+b)")
        
        plt.xlabel("Layer")
        plt.ylabel("Fit Quality (R²)")
        plt.title("Fit Quality of Helical Models Across Layers")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return results

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

if __name__ == "__main__":
    # Run experiments with a small model for demonstration
    experiments = HelicalExperiments(model_name="gpt2")
    
    # 1. Analyze Fourier structure of number representations
    print("Analyzing Fourier structure of number representations...")
    experiments.analyze_fourier_structure(max_num=100)
    
    # 2. Visualize the helical subspace
    print("Visualizing helical subspace...")
    experiments.visualize_helix_subspace(list(range(100)))
    
    # 3. Test addition mechanism
    print("Testing addition mechanism...")
    # Create test set of 10 simple addition problems
    a_values = [5, 12, 27, 31, 42, 50, 63, 71, 85, 93]
    b_values = [3, 8, 14, 22, 35, 48, 19, 26, 12, 5]
    experiments.analyze_addition_mechanism(a_values, b_values)
    
    print("Experiments complete!") 