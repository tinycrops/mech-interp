import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class HelicalEncoding(nn.Module):
    """
    Helical encoding for representing numbers in LLMs.
    Maps scalar values to points on a helix embedded in a hypersphere.
    """
    def __init__(self, dim, periods=None, learnable_periods=True):
        super().__init__()
        self.dim = dim
        self.num_helices = dim // 2  # Each helix needs 2 dimensions
        
        if periods is None:
            # Set default periods based on common numerical patterns
            self.periods = [10, 100, 24, 60, 7, 365, 12, 1000][:self.num_helices]
            # Repeat the last period if we need more
            self.periods += [1000] * (self.num_helices - len(self.periods))
        else:
            self.periods = periods[:self.num_helices]
        
        # Create learnable period parameters if requested
        if learnable_periods:
            self.period_params = nn.Parameter(torch.tensor(self.periods, dtype=torch.float))
        else:
            self.register_buffer('period_params', torch.tensor(self.periods, dtype=torch.float))
    
    def forward(self, x):
        """
        Maps scalar inputs to helical coordinates.
        Input: x - Tensor of shape (...) containing scalar values
        Output: Tensor of shape (..., dim) containing helical coordinates
        """
        # Reshape x for broadcasting
        x_expanded = x.unsqueeze(-1)  # Shape: (..., 1)
        
        # Compute angular position for each helix (phase)
        phases = 2 * math.pi * x_expanded / self.period_params  # Shape: (..., num_helices)
        
        # Map to 2D coordinates on each helix
        cos_vals = torch.cos(phases)  # Shape: (..., num_helices)
        sin_vals = torch.sin(phases)  # Shape: (..., num_helices)
        
        # Interleave cos and sin values to get final embedding
        result = torch.zeros(*x.shape, self.dim, device=x.device)
        result[..., 0::2] = cos_vals
        result[..., 1::2] = sin_vals
        
        return result

class HelicalAttention(nn.Module):
    """
    Custom attention mechanism optimized for helical representations.
    Enables number-aware attention for numerical reasoning in LLMs.
    """
    def __init__(self, embed_dim, num_heads, helix_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Optional helical projection dimension
        self.helix_dim = helix_dim or (self.head_dim // 2) * 2  # Ensure even number
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Helical projection layers
        self.helical_projections = nn.ModuleList([
            nn.Linear(self.head_dim, self.helix_dim) for _ in range(num_heads)
        ])
        
        # Period parameters for helical attention
        self.base_periods = [10, 100, 24, 60, 7, 365, 12, 1000]
        num_periods_needed = min(self.helix_dim // 2, len(self.base_periods))
        self.periods = nn.Parameter(
            torch.tensor(self.base_periods[:num_periods_needed], dtype=torch.float)
        )
    
    def _compute_helical_attention(self, q, k):
        """
        Compute attention scores using helical distance.
        Measures similarity in both the linear and circular components.
        """
        batch_size, seq_len, _ = q.shape
        
        # Reshape q and k for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Project to helical space for each head
        q_helical = []
        k_helical = []
        
        for head_idx in range(self.num_heads):
            q_head = q[:, :, head_idx]  # (batch_size, seq_len, head_dim)
            k_head = k[:, :, head_idx]  # (batch_size, seq_len, head_dim)
            
            # Project to helical dimensions
            q_proj = self.helical_projections[head_idx](q_head)  # (batch_size, seq_len, helix_dim)
            k_proj = self.helical_projections[head_idx](k_head)  # (batch_size, seq_len, helix_dim)
            
            q_helical.append(q_proj)
            k_helical.append(k_proj)
        
        # Stack helical projections
        q_helical = torch.stack(q_helical, dim=2)  # (batch_size, seq_len, num_heads, helix_dim)
        k_helical = torch.stack(k_helical, dim=2)  # (batch_size, seq_len, num_heads, helix_dim)
        
        # Compute dot product attention with helical awareness
        # For helical coordinates, we want to measure similarity while respecting periodicity
        
        # Standard dot product for the raw attention weight component
        # Reshape for proper batch matrix multiplication
        q_reshaped = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k_reshaped = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        dot_attn = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Result shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Helical similarity for each head (using cosine similarity on helical coordinates)
        helical_attn = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=q.device)
        
        for i in range(self.num_heads):
            q_h = q_helical[:, :, i]  # (batch_size, seq_len, helix_dim)
            k_h = k_helical[:, :, i]  # (batch_size, seq_len, helix_dim)
            
            # Reshape for batch matrix multiplication
            q_h = q_h.reshape(batch_size, seq_len, self.helix_dim // 2, 2)
            k_h = k_h.reshape(batch_size, seq_len, self.helix_dim // 2, 2)
            
            # Compute helical similarity (respecting the circular nature)
            # For each pair of 2D points on each helix
            for j in range(self.helix_dim // 2):
                # Get 2D coordinates on this helix
                q_coords = q_h[:, :, j]  # (batch_size, seq_len, 2)
                k_coords = k_h[:, :, j]  # (batch_size, seq_len, 2)
                
                # Compute dot product between normalized coordinates 
                # (resembles cosine of angle difference)
                q_norm = F.normalize(q_coords, p=2, dim=-1)  # Unit vectors
                k_norm = F.normalize(k_coords, p=2, dim=-1)  # Unit vectors
                
                # Reshape for proper matrix multiplication
                q_norm = q_norm.transpose(1, 2).reshape(batch_size, 1, seq_len, 2)
                k_norm = k_norm.transpose(1, 2).reshape(batch_size, 1, seq_len, 2)
                
                # Batch matrix multiplication to get similarity scores
                similarity = torch.matmul(q_norm, k_norm.transpose(-2, -1))  # (batch_size, 1, seq_len, seq_len)
                similarity = similarity.squeeze(1)  # (batch_size, seq_len, seq_len)
                
                # Add to the attention scores for this head
                helical_attn[:, i] += similarity.squeeze(-3)
            
            # Average over all helices
            helical_attn[:, i] /= (self.helix_dim // 2)
        
        # Combine standard attention with helical attention
        # Make sure shapes match for addition
        # (batch_size, num_heads, seq_len, seq_len) for both dot_attn and helical_attn
        
        # Weighted combination of both attention mechanisms
        # We use a learned balance parameter
        alpha = 0.5  # This could be a learned parameter
        combined_attn = alpha * dot_attn + (1 - alpha) * helical_attn
        
        return combined_attn
    
    def forward(self, x):
        """
        Forward pass for helical attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention scores with helical awareness
        attn_scores = self._compute_helical_attention(q, k)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Reshape v for attention application
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output

class ToroidalLayer(nn.Module):
    """
    Toroidal layer for representing multi-dimensional periodic data.
    Useful for capturing multiple interacting periodic patterns in LLMs.
    """
    def __init__(self, input_dim, output_dim, num_tori=4, periods=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tori = num_tori
        
        # Each torus is represented by 2 dimensions (sin & cos for each circle)
        self.torus_dim = 2 * num_tori
        
        # Period parameters for each circular dimension
        if periods is None:
            # Default periods for common patterns
            self.base_periods = [10, 100, 24, 60, 7, 365, 12, 1000][:num_tori]
            self.base_periods += [1000] * (num_tori - len(self.base_periods))
        else:
            self.base_periods = periods[:num_tori]
        
        self.periods = nn.Parameter(torch.tensor(self.base_periods, dtype=torch.float))
        
        # Input projection to get values to map to tori
        self.input_proj = nn.Linear(input_dim, num_tori)
        
        # Output projection from toroidal space to output space
        self.output_proj = nn.Linear(self.torus_dim, output_dim)
    
    def forward(self, x):
        """
        Map inputs to toroidal space and then to output space.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Project input to values for each toroidal dimension
        torus_values = self.input_proj(x)  # Shape: (..., num_tori)
        
        # Convert values to angular positions on each torus
        phases = 2 * math.pi * torus_values / self.periods.view(1, -1)
        
        # Map to coordinates on each torus
        cos_vals = torch.cos(phases)  # Shape: (..., num_tori)
        sin_vals = torch.sin(phases)  # Shape: (..., num_tori)
        
        # Interleave cos and sin values
        torus_coords = torch.zeros(*x.shape[:-1], self.torus_dim, device=x.device)
        torus_coords[..., 0::2] = cos_vals
        torus_coords[..., 1::2] = sin_vals
        
        # Project to output space
        output = self.output_proj(torus_coords)
        
        return output

class HelicalTransformerLayer(nn.Module):
    """
    Transformer layer enhanced with helical attention.
    Can be integrated into existing LLM architectures.
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super().__init__()
        
        # Helical attention
        self.attention = HelicalAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads
        )
        
        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        
        # Layer normalization
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states):
        # Self-attention with helical awareness
        attention_output = self.attention(hidden_states)
        attention_output = self.dropout(attention_output)
        
        # First residual connection and layer normalization
        hidden_states = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward network
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # Second residual connection and layer normalization
        layer_output = self.output_layer_norm(hidden_states + layer_output)
        
        return layer_output

class HypersphericalEmbedding(nn.Module):
    """
    Embedding layer that maps tokens to points on a hypersphere.
    Enables better representation of semantic relationships through angular distances.
    """
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embeddings on the hypersphere
        embedding = torch.randn(vocab_size, embed_dim)
        # Normalize to unit vectors
        embedding = F.normalize(embedding, p=2, dim=1)
        
        self.embedding = nn.Parameter(embedding)
        
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding[padding_idx].fill_(0)
    
    def forward(self, input_ids):
        """
        Maps token IDs to hyperspherical embeddings.
        
        Args:
            input_ids: Tensor of token IDs
            
        Returns:
            Hyperspherical embeddings for the tokens
        """
        # Get embeddings
        embeddings = F.embedding(input_ids, self.embedding)
        
        # Ensure embeddings are on the unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings

class NumberAwareTransformer(nn.Module):
    """
    Transformer model with specialized components for numerical reasoning.
    Combines hyperspherical embeddings, helical encodings, and toroidal layers.
    """
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout_prob=0.1,
        max_position_embeddings=512
    ):
        super().__init__()
        self.hidden_size = hidden_size  # Store hidden_size as instance variable
        
        # Token embeddings on hypersphere
        self.token_embeddings = HypersphericalEmbedding(vocab_size, hidden_size)
        
        # Position encoding with helical components
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.helical_encoding = HelicalEncoding(hidden_size // 4)  # Use 1/4 of dimensions for helical encoding
        
        # Toroidal layer for enhanced numerical representation
        self.toroidal_layer = ToroidalLayer(hidden_size, hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Transformer layers with helical attention
        self.layers = nn.ModuleList([
            HelicalTransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout_prob=dropout_prob
            )
            for _ in range(num_layers)
        ])
        
        # Final projection to vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, position_ids=None):
        batch_size, seq_length = input_ids.shape
        
        # Get hyperspherical token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get standard position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Get helical position encoding
        helical_pos = self.helical_encoding(position_ids.float())
        
        # Scale and reshape helical encoding to match hidden size
        helical_pos = helical_pos.view(batch_size, seq_length, -1)
        helical_pos_padded = torch.zeros(
            batch_size, seq_length, self.hidden_size, device=input_ids.device
        )
        helical_pos_padded[:, :, :helical_pos.size(-1)] = helical_pos
        
        # Combine embeddings with helical position encoding
        hidden_states = token_embeddings + position_embeddings + helical_pos_padded
        
        # Apply toroidal transformation for enhanced numerical representation
        toroidal_output = self.toroidal_layer(hidden_states)
        
        # Combine original embeddings with toroidal output
        # Use a residual connection to preserve original embeddings
        hidden_states = hidden_states + 0.1 * toroidal_output
        
        # Layer normalization and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final projection to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits

class HelicalMLP(nn.Module):
    """
    A multi-layer perceptron that incorporates helical representations for improved numerical reasoning.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, periods=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Layers for main path
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.main_path = nn.Sequential(*layers)
        
        # Helical encoding path
        self.helical_encoding = HelicalEncoding(input_dim * 2, periods=periods)
        self.helical_projection = nn.Linear(input_dim * 2, output_dim)
    
    def forward(self, x):
        # Main path
        main_output = self.main_path(x)
        
        # Helical path (for numerical features)
        helical_repr = self.helical_encoding(x)
        helical_output = self.helical_projection(helical_repr)
        
        # Combine outputs (weighted sum)
        alpha = 0.7  # Could be a learned parameter
        combined_output = alpha * main_output + (1 - alpha) * helical_output
        
        return combined_output

# Visualization utilities for analyzing helical representations

def visualize_helical_space(model, token_ids, position_ids=None):
    """
    Extracts and visualizes the helical representation of tokens.
    
    Args:
        model: The LLM with helical components
        token_ids: Input token IDs
        position_ids: Optional position IDs
        
    Returns:
        Dictionary of extracted helical components for visualization
    """
    # Extract helical representations from the model
    # This is a template that would need to be adapted to the specific model
    
    # Placeholder implementation
    batch_size, seq_length = token_ids.shape
    
    # Generate position IDs if not provided
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # Extract the helical encoding for the positions
    helical_encoding = model.helical_encoding(position_ids.float())
    
    # Extract attention patterns from the model
    # This would depend on how the model is implemented
    
    return {
        'helical_encoding': helical_encoding.detach().cpu().numpy(),
        # Additional components would be added here
    }

# Example of how to use the helical number extractor
def extract_numerical_values(model, token_ids):
    """
    Extracts numerical values from the helical representations in the model.
    
    Args:
        model: The LLM with helical components
        token_ids: Input token IDs
        
    Returns:
        Tensor of extracted numerical values
    """
    # Get model representations
    # This is a placeholder implementation that would need adaptation
    
    # Run model in evaluation mode
    model.eval()
    with torch.no_grad():
        # Get embeddings and helical encodings
        token_embeddings = model.token_embeddings(token_ids)
        
        # For demonstration, assuming a specific layer exposes helical values
        # In a real implementation, we'd need to extract from the appropriate layers
        
        # Apply an inverse transformation to get numerical values
        # This is a simplified example
        extracted_values = model.extract_numerical_values(token_embeddings)
    
    return extracted_values 