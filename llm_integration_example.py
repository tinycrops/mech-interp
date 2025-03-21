import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from helical_llm_models import (
    HelicalEncoding, 
    HelicalAttention, 
    ToroidalLayer, 
    NumberAwareTransformer,
    HelicalTransformerLayer
)

# This example shows how to enhance an existing LLM with helical components
# for improved numerical reasoning

class EnhancedLLMForNumericalReasoning:
    """
    Wrapper class to enhance an existing LLM with helical components
    for improved numerical reasoning.
    """
    def __init__(self, base_llm, hidden_size=768):
        """
        Initialize the enhanced LLM.
        
        Args:
            base_llm: The original LLM (could be a HuggingFace model)
            hidden_size: Hidden size of the model
        """
        self.base_llm = base_llm
        self.hidden_size = hidden_size
        
        # Add specialized components for numerical reasoning
        self.num_detector = self._create_number_detector()
        self.helical_encoder = HelicalEncoding(dim=hidden_size // 2)
        self.toroidal_layer = ToroidalLayer(hidden_size, hidden_size)
        
        # Helical attention layer for handling numerical contexts
        self.helical_attention = HelicalAttention(
            embed_dim=hidden_size,
            num_heads=12
        )
        
        # Projection layers
        self.pre_helical_proj = nn.Linear(hidden_size, hidden_size)
        self.post_helical_proj = nn.Linear(hidden_size, hidden_size)
    
    def _create_number_detector(self):
        """
        Create a simple detector that identifies whether a token 
        represents a numerical value.
        
        Returns:
            A function that estimates the "numberness" of a token
        """
        # In practice, this would be a small neural network
        # For simplicity, we'll use a dummy implementation
        def is_number_token(token_embedding):
            # This is a placeholder - in a real implementation,
            # we would train a network to detect numerical tokens
            return torch.ones_like(token_embedding[:, :, 0])
        
        return is_number_token
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the enhanced LLM.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Model outputs with enhanced numerical reasoning
        """
        # Get base model embeddings
        base_outputs = self.base_llm.get_input_embeddings()(input_ids)
        hidden_states = base_outputs
        
        # Detect numerical tokens
        numberness = self.num_detector(hidden_states)  # [batch_size, seq_len]
        numberness = numberness.unsqueeze(-1)  # Add feature dimension for broadcasting
        
        # Process with helical components for numerical tokens
        helical_hidden = self.pre_helical_proj(hidden_states)
        
        # Generate numerical values from embeddings (simplified)
        batch_size, seq_len = input_ids.shape
        # This would be more sophisticated in a real implementation
        # Here we're just using token IDs as placeholder numerical values
        numerical_values = input_ids.float() % 1000  # Limit to reasonable range
        
        # Apply helical encoding to numerical values
        helical_encoded = self.helical_encoder(numerical_values)
        
        # Reshape helical encoding to match hidden size
        helical_encoded_padded = torch.zeros_like(hidden_states)
        helical_encoded = helical_encoded.view(batch_size, seq_len, -1)
        helical_encoded_padded[:, :, :helical_encoded.size(-1)] = helical_encoded
        
        # Combine original hidden states with helical encoding based on numberness
        enhanced_hidden = hidden_states + numberness * helical_encoded_padded
        
        # Apply toroidal layer for multi-dimensional periodicity
        toroidal_output = self.toroidal_layer(enhanced_hidden)
        enhanced_hidden = enhanced_hidden + numberness * 0.1 * toroidal_output
        
        # Apply helical attention for enhanced numerical reasoning
        helical_attention_output = self.helical_attention(enhanced_hidden)
        
        # Blend original and helical attention based on numberness
        blended_hidden = (1 - numberness) * hidden_states + numberness * helical_attention_output
        
        # Project back to original hidden space
        final_hidden = self.post_helical_proj(blended_hidden)
        
        # Continue with the base LLM's processing
        # In practice, you would insert this into the appropriate layer
        # of the base model's processing pipeline
        
        # This is a simplified example - the actual integration would depend
        # on the specific LLM architecture
        
        return final_hidden


# Demonstration of fine-tuning with numerical reasoning tasks

def create_numerical_reasoning_dataset():
    """
    Create a synthetic dataset for numerical reasoning.
    
    Returns:
        Inputs and targets for a simple arithmetic task
    """
    # Generate 1000 simple addition problems
    num_samples = 1000
    max_num = 100
    
    # Create random numbers
    a = np.random.randint(1, max_num, size=num_samples)
    b = np.random.randint(1, max_num, size=num_samples)
    
    # Compute sums
    sums = a + b
    
    # Create simple token sequences: [a, +, b, =, sum]
    # Using token IDs directly for simplicity
    inputs = torch.zeros((num_samples, 4), dtype=torch.long)
    inputs[:, 0] = torch.tensor(a)
    inputs[:, 1] = torch.tensor(max_num + 1)  # "+" token
    inputs[:, 2] = torch.tensor(b)
    inputs[:, 3] = torch.tensor(max_num + 2)  # "=" token
    
    # Targets are the sums
    targets = torch.tensor(sums, dtype=torch.long)
    
    return inputs, targets


def fine_tune_numerical_reasoning():
    """
    Fine-tune a model on numerical reasoning tasks.
    Compare standard vs. helical-enhanced models.
    """
    # Create dataset
    inputs, targets = create_numerical_reasoning_dataset()
    
    # Split data
    split = int(0.8 * len(inputs))
    train_inputs, val_inputs = inputs[:split], inputs[split:]
    train_targets, val_targets = targets[:split], targets[split:]
    
    # Create models
    vocab_size = 200  # Large enough for our synthetic task
    hidden_size = 128
    
    # Standard model
    standard_model = nn.Sequential(
        nn.Embedding(vocab_size, hidden_size),
        nn.Flatten(),
        nn.Linear(4 * hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 200)  # Output up to sums of 200
    )
    
    # Helical model
    class HelicalModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.helical_encoding = HelicalEncoding(hidden_size // 2)
            self.flatten = nn.Flatten()
            self.hidden = nn.Linear(4 * hidden_size, hidden_size)
            self.activation = nn.ReLU()
            self.output = nn.Linear(hidden_size, 200)
        
        def forward(self, x):
            # Standard embedding path
            embedded = self.embedding(x)
            
            # Helical path for numerical values
            # Treat token IDs as numerical values for simplicity
            numerical_values = x.float()
            helical_encoded = self.helical_encoding(numerical_values)
            
            # Reshape to match embedding
            batch_size, seq_len = x.shape
            helical_padded = torch.zeros_like(embedded)
            helical_encoded = helical_encoded.view(batch_size, seq_len, -1)
            helical_padded[:, :, :helical_encoded.size(-1)] = helical_encoded
            
            # Add helical information to embeddings
            enhanced_embeddings = embedded + 0.1 * helical_padded
            
            # Process through the network
            flattened = self.flatten(enhanced_embeddings)
            hidden = self.hidden(flattened)
            activated = self.activation(hidden)
            output = self.output(activated)
            
            return output
    
    helical_model = HelicalModel(vocab_size, hidden_size)
    
    # Training loop
    def train_model(model, name):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        batch_size = 32
        num_epochs = 10
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            # Create batches
            indices = torch.randperm(len(train_inputs))
            for i in range(0, len(train_inputs), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_inputs = train_inputs[batch_indices]
                batch_targets = train_targets[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / (len(train_inputs) // batch_size)
            train_losses.append(epoch_loss)
            
            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                outputs = model(val_inputs)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == val_targets).sum().item()
            
            accuracy = correct / len(val_targets)
            val_accuracies.append(accuracy)
            
            print(f"{name} - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return train_losses, val_accuracies
    
    # Train both models
    standard_losses, standard_acc = train_model(standard_model, "Standard")
    helical_losses, helical_acc = train_model(helical_model, "Helical")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(standard_losses, label='Standard Model')
    plt.plot(helical_losses, label='Helical Model')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(standard_acc, label='Standard Model')
    plt.plot(helical_acc, label='Helical Model')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return standard_model, helical_model

# Demonstration of enhanced number representation in LLMs

def demonstrate_numerical_extraction():
    """
    Demonstrate how to extract numerical values from LLM representations.
    """
    # Create a simple model with helical components
    vocab_size = 1000  # Increased vocab size to accommodate our test numbers
    hidden_size = 64
    
    # Add this method to the NumberAwareTransformer class to fix the error
    # This is a monkey patch for demonstration purposes
    def extract_numerical_values(embeddings):
        """Extracts numerical values from embeddings based on their helical representation"""
        # In a real implementation, this would involve analyzing helical patterns
        # Here we'll just return placeholder values for demonstration
        batch_size, seq_len = embeddings.shape[:2]
        return torch.arange(seq_len).float().unsqueeze(0).expand(batch_size, -1)
        
    # Add the method to the NumberAwareTransformer class
    NumberAwareTransformer.extract_numerical_values = extract_numerical_values
    
    model = NumberAwareTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=2,
        num_attention_heads=2
    )
    
    # Generate a sequence of token IDs representing numbers
    # Make sure our numbers don't exceed the vocabulary size
    numbers = [5, 10, 42, 100, 365]
    token_ids = torch.tensor([numbers], dtype=torch.long)  # Add batch dimension
    
    # Get model representations
    with torch.no_grad():
        # Forward pass to get representations
        output = model(token_ids)
        
        # Extract the helical components
        # In a real implementation, we would need to analyze the model's representations
        # to find where and how the numerical values are encoded
        
        # For demonstration purposes, let's assume we've identified the relevant
        # dimensions in the model's internal representations
        
        # Extract helical position encoding
        position_ids = torch.arange(len(numbers)).unsqueeze(0)  # [1, seq_len]
        helical_encoding = model.helical_encoding(position_ids.float())
        
        # Apply reverse transformation to extract numerical values
        # In practice, this would involve more sophisticated analysis
        
        # Print the results
        print("Original numbers:", numbers)
        print("Model representations shape:", output.shape)
        print("Helical encoding shape:", helical_encoding.shape)
        
        # Visualization of how numbers are represented
        plt.figure(figsize=(10, 6))
        
        # Visualize the first two helical dimensions for each number
        for i, num in enumerate(numbers):
            # Extract the first helical encoding (first two dimensions)
            x = helical_encoding[0, i, 0].item()
            y = helical_encoding[0, i, 1].item()
            
            plt.scatter(x, y, s=100, label=f'Number {num}')
            plt.text(x+0.05, y+0.05, str(num))
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=0.3)
        plt.title("Helical Representation of Numbers in the LLM")
        plt.xlabel("First Dimension")
        plt.ylabel("Second Dimension")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Fine-tuning comparison between standard and helical models:")
    standard_model, helical_model = fine_tune_numerical_reasoning()
    
    print("\nDemonstrating numerical value extraction from LLM:")
    demonstrate_numerical_extraction() 