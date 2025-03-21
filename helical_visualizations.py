import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from helical_llm_models import HelicalEncoding, ToroidalLayer, NumberAwareTransformer

def plot_helix(periods, num_points=100, title="Helical Representation"):
    """
    Plot a 3D helix visualization of helical encoding.
    
    Args:
        periods: List of period values to visualize
        num_points: Number of points to plot
        title: Plot title
    """
    plt.figure(figsize=(15, 12))
    
    for i, period in enumerate(periods[:4]):  # Show up to 4 periods
        # Create 3D plot - fixing the projection error
        ax = plt.subplot(2, 2, i+1, projection='3d')
        
        # Generate points along the helix
        t = np.linspace(0, 3 * period, num_points)
        
        # Linear component (z-axis)
        z = t
        
        # Circular components (x,y axes)
        phase = 2 * np.pi * t / period
        x = np.cos(phase)
        y = np.sin(phase)
        
        # Color points by their numerical value
        colors = cm.viridis(t / max(t))
        
        # Plot the helix
        ax.scatter(x, y, z, c=colors, alpha=0.8)
        
        # Connect points with a line
        ax.plot(x, y, z, color='gray', alpha=0.3)
        
        # Highlight complete turns
        for turn in range(1, int(max(t) / period) + 1):
            turn_idx = np.argmin(abs(t - turn * period))
            ax.scatter([x[turn_idx]], [y[turn_idx]], [z[turn_idx]], 
                       color='red', s=100, marker='o')
        
        ax.set_title(f"Helix with Period = {period}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Value")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def visualize_helical_encoding():
    """
    Visualize how different numerical values are encoded on helices.
    """
    # Create a helical encoding layer
    encoding_dim = 8  # 4 helices (8 dimensions)
    periods = [10, 100, 60, 365]  # Common periodicities
    encoder = HelicalEncoding(encoding_dim, periods=periods, learnable_periods=False)
    
    # Generate a range of numbers to encode
    values = torch.linspace(0, 500, 1000)
    
    # Encode the values
    with torch.no_grad():
        encoded = encoder(values)
    
    # Convert to numpy for plotting
    encoded_np = encoded.numpy()
    values_np = values.numpy()
    
    # Visualize each helix (pairs of dimensions)
    plt.figure(figsize=(16, 12))
    
    for i in range(encoding_dim // 2):
        plt.subplot(2, 2, i+1)
        
        # Get the coordinates for this helix
        x_coords = encoded_np[:, i*2]
        y_coords = encoded_np[:, i*2 + 1]
        
        # Create a scatter plot colored by the original values
        scatter = plt.scatter(x_coords, y_coords, c=values_np, cmap='viridis', alpha=0.6)
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Numerical Value')
        
        # Plot the unit circle to show the constraint
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        # Mark specific values
        for val in [0, 50, 100, 200, 300, 400, 500]:
            idx = np.argmin(np.abs(values_np - val))
            plt.plot(x_coords[idx], y_coords[idx], 'ro', markersize=8)
            plt.text(x_coords[idx]*1.1, y_coords[idx]*1.1, str(val))
        
        plt.title(f"Helix {i+1} (Period = {periods[i]})")
        plt.xlabel("cos(2π × value / period)")
        plt.ylabel("sin(2π × value / period)")
        plt.grid(alpha=0.3)
        plt.axis('equal')
    
    plt.suptitle("Helical Encoding of Numerical Values", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def animate_helical_addition(a=5, b=7, period=10):
    """
    Create an animation demonstrating addition on a helix.
    
    Args:
        a, b: Numbers to add
        period: Period of the helix
    """
    # Create figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate the helix
    t = np.linspace(0, 3 * period, 300)
    phase = 2 * np.pi * t / period
    x = np.cos(phase)
    y = np.sin(phase)
    z = t
    
    # Plot the helix
    ax.plot(x, y, z, color='gray', alpha=0.5)
    
    # Find points for a, b, and a+b
    a_idx = np.argmin(np.abs(t - a))
    b_idx = np.argmin(np.abs(t - b))
    sum_idx = np.argmin(np.abs(t - (a + b)))
    
    # Create points for animation
    point_a = ax.scatter([x[a_idx]], [y[a_idx]], [z[a_idx]], color='blue', s=100, label=f'a = {a}')
    point_b = ax.scatter([x[b_idx]], [y[b_idx]], [z[b_idx]], color='green', s=100, label=f'b = {b}')
    point_sum = ax.scatter([x[sum_idx]], [y[sum_idx]], [z[sum_idx]], color='red', s=100, label=f'a + b = {a+b}')
    
    # Animation frames
    frames = 100
    
    # Define the update function for animation
    def update(frame):
        # Animate movement from a to a+b
        if frame < frames//2:
            # First half: translate vertically by b
            progress = frame / (frames//2)
            current_z = z[a_idx] + progress * b
            
            # Find the closest point on the helix
            closest_idx = np.argmin(np.abs(z - current_z))
            point_sum._offsets3d = ([x[closest_idx]], [y[closest_idx]], [z[closest_idx]])
        else:
            # Second half: show final result
            point_sum._offsets3d = ([x[sum_idx]], [y[sum_idx]], [z[sum_idx]])
        
        return point_a, point_b, point_sum
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    ax.set_title(f"Addition on a Helix (Period = {period})")
    ax.legend()
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    plt.tight_layout()
    return ani

def visualize_toroidal_layer():
    """
    Visualize a toroidal layer for representing multi-dimensional periodicity.
    """
    # Create a toroidal layer
    input_dim = 2
    output_dim = 8
    num_tori = 2
    periods = [24, 7]  # Hours in day, days in week
    
    torus_layer = ToroidalLayer(input_dim, output_dim, num_tori=num_tori, periods=periods)
    
    # Generate grid of input values
    hour = torch.linspace(0, 48, 49)  # Two days worth of hours
    day = torch.linspace(0, 14, 15)   # Two weeks worth of days
    
    hour_grid, day_grid = torch.meshgrid(hour, day)
    inputs = torch.stack([hour_grid.flatten(), day_grid.flatten()], dim=1)
    
    # Pass through the toroidal layer
    with torch.no_grad():
        outputs = torus_layer(inputs)
    
    # Extract the first torus (hours)
    hour_x = outputs[:, 0].reshape(hour.size(0), day.size(0)).numpy()
    hour_y = outputs[:, 1].reshape(hour.size(0), day.size(0)).numpy()
    
    # Extract the second torus (days)
    day_x = outputs[:, 2].reshape(hour.size(0), day.size(0)).numpy()
    day_y = outputs[:, 3].reshape(hour.size(0), day.size(0)).numpy()
    
    # Visualize
    plt.figure(figsize=(16, 8))
    
    # Hour torus
    plt.subplot(1, 2, 1)
    plt.pcolormesh(hour.numpy(), day.numpy(), hour_x.T, shading='auto', cmap='viridis')
    plt.colorbar(label='cos(2π × hour / 24)')
    plt.title("Hour Periodicity (X coordinate)")
    plt.xlabel("Hour")
    plt.ylabel("Day")
    
    # Day torus
    plt.subplot(1, 2, 2)
    plt.pcolormesh(hour.numpy(), day.numpy(), day_x.T, shading='auto', cmap='viridis')
    plt.colorbar(label='cos(2π × day / 7)')
    plt.title("Day Periodicity (X coordinate)")
    plt.xlabel("Hour")
    plt.ylabel("Day")
    
    plt.suptitle("Toroidal Representation of Time", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def compare_attention_patterns():
    """
    Compare standard attention vs. helical attention for numerical sequences.
    """
    # Create toy sequences with numerical patterns
    # We'll use a simple counting pattern and a periodic pattern
    counting_seq = torch.tensor([i for i in range(10)]).float().unsqueeze(0)
    periodic_seq = torch.tensor([i % 7 for i in range(10)]).float().unsqueeze(0)
    
    # Create a toy transformer with helical attention
    vocab_size = 20  # Simplified vocabulary
    model = NumberAwareTransformer(
        vocab_size=vocab_size,
        hidden_size=32,
        num_layers=1,
        num_attention_heads=2
    )
    
    # Access the helical attention directly for visualization
    helical_attention = model.layers[0].attention
    
    # Create input token IDs from our sequences
    # For this demo, we'll just use the numerical values as token IDs
    counting_tokens = counting_seq.long()
    periodic_tokens = periodic_seq.long()
    
    # Get embeddings
    with torch.no_grad():
        counting_embeddings = model.token_embeddings(counting_tokens)
        periodic_embeddings = model.token_embeddings(periodic_tokens)
        
        # Compute attention scores
        # We'll modify the attention module to return attention scores for visualization
        q_counting = helical_attention.q_proj(counting_embeddings)
        k_counting = helical_attention.k_proj(counting_embeddings)
        counting_attention = helical_attention._compute_helical_attention(q_counting, k_counting)
        
        q_periodic = helical_attention.q_proj(periodic_embeddings)
        k_periodic = helical_attention.k_proj(periodic_embeddings)
        periodic_attention = helical_attention._compute_helical_attention(q_periodic, k_periodic)
    
    # Visualize attention patterns
    plt.figure(figsize=(12, 10))
    
    # Counting sequence attention
    plt.subplot(2, 2, 1)
    plt.imshow(counting_attention[0, 0].numpy(), cmap='viridis')
    plt.colorbar(label='Attention Score')
    plt.title("Helical Attention - Counting Sequence (Head 1)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    
    plt.subplot(2, 2, 2)
    plt.imshow(counting_attention[0, 1].numpy(), cmap='viridis')
    plt.colorbar(label='Attention Score')
    plt.title("Helical Attention - Counting Sequence (Head 2)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    
    # Periodic sequence attention
    plt.subplot(2, 2, 3)
    plt.imshow(periodic_attention[0, 0].numpy(), cmap='viridis')
    plt.colorbar(label='Attention Score')
    plt.title("Helical Attention - Periodic Sequence (Head 1)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    
    plt.subplot(2, 2, 4)
    plt.imshow(periodic_attention[0, 1].numpy(), cmap='viridis')
    plt.colorbar(label='Attention Score')
    plt.title("Helical Attention - Periodic Sequence (Head 2)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    
    plt.suptitle("Comparison of Attention Patterns for Numerical Sequences", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run visualizations
    plot_helix([10, 100, 24, 365])
    visualize_helical_encoding()
    ani = animate_helical_addition(a=5, b=7, period=10)
    plt.show()
    visualize_toroidal_layer()
    compare_attention_patterns() 