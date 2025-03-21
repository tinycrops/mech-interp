import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math
from helical_llm_models import HelicalEncoding

class ClockAlgorithmVisualization:
    """
    Visualizes the "Clock Algorithm" described in the paper
    "Language Models Use Trigonometry to Do Addition"
    """
    def __init__(self):
        # Define periods for helical representations
        self.periods = [10, 100]  # Focus on base-10 system
        
        # Initialize helical encoding
        self.encoding = HelicalEncoding(dim=4, periods=self.periods, learnable_periods=False)
    
    def plot_number_helix(self, num_range=(0, 99), period=10, animate=False):
        """
        Plot a 3D helix showing how numbers are represented.
        
        Args:
            num_range: Range of numbers to represent
            period: Primary period to visualize
            animate: Whether to create an animation
        """
        numbers = np.arange(num_range[0], num_range[1] + 1)
        
        # Convert to tensor
        numbers_tensor = torch.tensor(numbers, dtype=torch.float32)
        
        # Get helical encoding
        with torch.no_grad():
            encoded = self.encoding(numbers_tensor)
        
        # Extract coordinates for the selected period
        period_idx = self.periods.index(period) if period in self.periods else 0
        x = encoded[:, period_idx*2].numpy()  # cos component
        y = encoded[:, period_idx*2 + 1].numpy()  # sin component
        z = numbers  # Linear component
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the helix
        if not animate:
            # Color points by their numerical value
            scatter = ax.scatter(x, y, z, c=numbers, cmap='viridis', s=30)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Number Value')
            
            # Connect points with a line
            ax.plot(x, y, z, color='gray', alpha=0.3)
            
            # Highlight complete turns
            for turn in range(int(num_range[0]/period), int(num_range[1]/period) + 1):
                # Find the closest point to turn*period
                turn_value = turn * period
                if turn_value >= num_range[0] and turn_value <= num_range[1]:
                    idx = np.argmin(np.abs(numbers - turn_value))
                    ax.scatter([x[idx]], [y[idx]], [z[idx]], color='red', s=100, marker='o')
                    ax.text(x[idx], y[idx], z[idx], f"{turn_value}", fontsize=10)
            
            # Set labels and title
            ax.set_xlabel('cos(2π × n / period)')
            ax.set_ylabel('sin(2π × n / period)')
            ax.set_zlabel('Number Value')
            ax.set_title(f"Helical Representation of Numbers (Period = {period})")
            
            plt.tight_layout()
            plt.show()
        else:
            # Create animation
            line, = ax.plot([], [], [], color='blue', lw=2)
            point = ax.scatter([], [], [], color='red', s=100)
            text = ax.text(0, 0, 0, "", fontsize=12)
            
            # Set initial view
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([num_range[0], num_range[1]])
            
            # Set labels and title
            ax.set_xlabel('cos(2π × n / period)')
            ax.set_ylabel('sin(2π × n / period)')
            ax.set_zlabel('Number Value')
            ax.set_title(f"Helical Representation of Numbers (Period = {period})")
            
            # Animation update function
            def update(frame):
                idx = frame % len(numbers)
                line.set_data(x[:idx+1], y[:idx+1])
                line.set_3d_properties(z[:idx+1])
                point._offsets3d = ([x[idx]], [y[idx]], [z[idx]])
                text.set_position((x[idx], y[idx]))
                text.set_3d_properties(z[idx])
                text.set_text(f"{numbers[idx]}")
                return line, point, text
            
            ani = FuncAnimation(fig, update, frames=len(numbers), interval=50, blit=True)
            plt.tight_layout()
            plt.show()
            
            return ani
    
    def visualize_addition_on_helix(self, a, b, period=10):
        """
        Visualize how addition works on a helix using the Clock algorithm.
        
        Args:
            a, b: Numbers to add
            period: Period of the helix to visualize
        """
        # Generate numbers for the helix
        numbers = np.arange(0, max(a+b, 100))
        
        # Convert to tensor
        numbers_tensor = torch.tensor(numbers, dtype=torch.float32)
        
        # Get helical encoding
        with torch.no_grad():
            encoded = self.encoding(numbers_tensor)
        
        # Extract coordinates for the selected period
        period_idx = self.periods.index(period) if period in self.periods else 0
        x = encoded[:, period_idx*2].numpy()  # cos component
        y = encoded[:, period_idx*2 + 1].numpy()  # sin component
        z = numbers  # Linear component
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the helix
        ax.plot(x, y, z, color='gray', alpha=0.5)
        
        # Find indices for a, b, and a+b
        a_idx = np.where(numbers == a)[0][0]
        b_idx = np.where(numbers == b)[0][0]
        sum_idx = np.where(numbers == a+b)[0][0]
        
        # Plot points a, b, and a+b
        ax.scatter(x[a_idx], y[a_idx], z[a_idx], color='blue', s=100, label=f'a = {a}')
        ax.scatter(x[b_idx], y[b_idx], z[b_idx], color='green', s=100, label=f'b = {b}')
        ax.scatter(x[sum_idx], y[sum_idx], z[sum_idx], color='red', s=100, label=f'a + b = {a+b}')
        
        # Calculate intermediate point for visualization (a rotated by b)
        angle_a = 2 * math.pi * a / period
        angle_b = 2 * math.pi * b / period
        angle_sum = angle_a + angle_b
        
        # Normalize angle_sum to [0, 2π)
        angle_sum = angle_sum % (2 * math.pi)
        
        # Calculate coordinates of intermediate point at height a
        x_intermediate = math.cos(angle_sum)
        y_intermediate = math.sin(angle_sum)
        
        # Plot intermediate point and connect with arrows
        ax.scatter([x_intermediate], [y_intermediate], [a], color='purple', s=80, 
                  label='Rotated Position', alpha=0.7)
        
        # Add curved arrow to show rotation from a by angle_b
        # This is a simplified representation of the rotation
        theta = np.linspace(angle_a, angle_sum, 30)
        x_curve = np.cos(theta)
        y_curve = np.sin(theta)
        z_curve = np.ones_like(theta) * a
        ax.plot(x_curve, y_curve, z_curve, 'r--', alpha=0.6)
        
        # Add arrow from intermediate point to final sum
        ax.plot([x_intermediate, x[sum_idx]], [y_intermediate, y[sum_idx]], [a, a+b], 
                'b--', alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('cos(2π × n / period)')
        ax.set_ylabel('sin(2π × n / period)')
        ax.set_zlabel('Number Value')
        ax.set_title(f"Clock Algorithm: {a} + {b} = {a+b} (Period = {period})")
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_animation_of_addition(self, a, b, period=10):
        """
        Create an animation showing the Clock algorithm in action.
        
        Args:
            a, b: Numbers to add
            period: Period of the helix to visualize
            
        Returns:
            Animation object
        """
        # Generate numbers for the helix
        numbers = np.arange(0, max(a+b, 100))
        
        # Convert to tensor
        numbers_tensor = torch.tensor(numbers, dtype=torch.float32)
        
        # Get helical encoding
        with torch.no_grad():
            encoded = self.encoding(numbers_tensor)
        
        # Extract coordinates for the selected period
        period_idx = self.periods.index(period) if period in self.periods else 0
        x = encoded[:, period_idx*2].numpy()  # cos component
        y = encoded[:, period_idx*2 + 1].numpy()  # sin component
        z = numbers  # Linear component
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the helix
        ax.plot(x, y, z, color='gray', alpha=0.3)
        
        # Set view limits
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([0, a+b+10])
        
        # Find indices for a, b, and a+b
        a_idx = np.where(numbers == a)[0][0]
        b_idx = np.where(numbers == b)[0][0]
        sum_idx = np.where(numbers == a+b)[0][0]
        
        # Create points for animation
        point_a = ax.scatter([x[a_idx]], [y[a_idx]], [z[a_idx]], color='blue', s=100, label=f'a = {a}')
        point_b = ax.scatter([x[b_idx]], [y[b_idx]], [z[b_idx]], color='green', s=100, label=f'b = {b}')
        point_sum = ax.scatter([], [], [], color='red', s=100, label=f'a + b = {a+b}')
        
        # Add text labels
        text_a = ax.text(x[a_idx], y[a_idx], z[a_idx], f"a = {a}", fontsize=10)
        text_b = ax.text(x[b_idx], y[b_idx], z[b_idx], f"b = {b}", fontsize=10)
        text_sum = ax.text(0, 0, 0, "", fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('cos(2π × n / period)')
        ax.set_ylabel('sin(2π × n / period)')
        ax.set_zlabel('Number Value')
        ax.set_title(f"Clock Algorithm: {a} + {b} = {a+b} (Period = {period})")
        ax.legend()
        
        # Animation functions
        frames = 60
        
        def update(frame):
            if frame < frames//3:
                # First part: show original a and b
                progress = frame / (frames//3)
                ax.view_init(elev=30, azim=progress * 90)
                text_sum.set_text("")
                return point_a, point_b, point_sum, text_a, text_b, text_sum
            
            elif frame < 2*frames//3:
                # Second part: rotate by angle_b at height a
                progress = (frame - frames//3) / (frames//3)
                
                # Calculate angle of rotation
                angle_a = 2 * math.pi * a / period
                angle_b = 2 * math.pi * b / period
                current_angle = angle_a + progress * angle_b
                
                # Calculate position
                x_rot = math.cos(current_angle)
                y_rot = math.sin(current_angle)
                z_rot = a
                
                # Update position
                point_sum._offsets3d = ([x_rot], [y_rot], [z_rot])
                text_sum.set_position((x_rot, y_rot))
                text_sum.set_3d_properties(z_rot)
                text_sum.set_text(f"Rotating: {progress:.1f} * {b}")
                
                return point_a, point_b, point_sum, text_a, text_b, text_sum
            
            else:
                # Third part: move up by b to final position
                progress = (frame - 2*frames//3) / (frames//3)
                
                # Calculate angular position
                angle_sum = 2 * math.pi * (a + b) / period
                x_sum = math.cos(angle_sum)
                y_sum = math.sin(angle_sum)
                
                # Calculate vertical position (move from a to a+b)
                z_sum = a + progress * b
                
                # Update position
                point_sum._offsets3d = ([x_sum], [y_sum], [z_sum])
                text_sum.set_position((x_sum, y_sum))
                text_sum.set_3d_properties(z_sum)
                text_sum.set_text(f"a + b = {a+b}")
                
                return point_a, point_b, point_sum, text_a, text_b, text_sum
        
        ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        plt.tight_layout()
        plt.show()
        
        return ani
    
    def visualize_helix_projection(self, period=10):
        """
        Visualize how the helix projects to a circle when viewed from above.
        
        Args:
            period: Period of the helix to visualize
        """
        # Generate numbers for the helix
        numbers = np.arange(0, 100)
        
        # Convert to tensor
        numbers_tensor = torch.tensor(numbers, dtype=torch.float32)
        
        # Get helical encoding
        with torch.no_grad():
            encoded = self.encoding(numbers_tensor)
        
        # Extract coordinates for the selected period
        period_idx = self.periods.index(period) if period in self.periods else 0
        x = encoded[:, period_idx*2].numpy()  # cos component
        y = encoded[:, period_idx*2 + 1].numpy()  # sin component
        z = numbers  # Linear component
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        
        # 3D helix
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x, y, z, color='gray', alpha=0.5)
        ax1.scatter(x, y, z, c=numbers, cmap='viridis')
        ax1.set_xlabel('cos(2π × n / period)')
        ax1.set_ylabel('sin(2π × n / period)')
        ax1.set_zlabel('Number Value')
        ax1.set_title(f"3D Helical Representation (Period = {period})")
        
        # 2D circle (top view)
        ax2 = fig.add_subplot(122)
        scatter = ax2.scatter(x, y, c=numbers, cmap='viridis')
        
        # Add the unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        # Highlight specific numbers
        for num in [0, period//2, period, period*1.5, period*2]:
            if num < 100:
                idx = np.where(numbers == num)[0][0]
                ax2.plot(x[idx], y[idx], 'ro', markersize=8)
                ax2.text(x[idx]*1.1, y[idx]*1.1, str(int(num)))
        
        ax2.set_xlabel('cos(2π × n / period)')
        ax2.set_ylabel('sin(2π × n / period)')
        ax2.set_title(f"2D Circular Projection (Period = {period})")
        ax2.axis('equal')
        plt.colorbar(scatter, ax=ax2, label='Number Value')
        
        plt.tight_layout()
        plt.show()

    def demonstrate_clock_algorithm(self):
        """
        Run a demonstration of the Clock algorithm with multiple examples.
        """
        # Example 1: Simple addition within period
        print("Example 1: Simple addition within period (5 + 3 = 8)")
        self.visualize_addition_on_helix(5, 3, period=10)
        
        # Example 2: Addition with carry
        print("Example 2: Addition with carry (7 + 5 = 12)")
        self.visualize_addition_on_helix(7, 5, period=10)
        
        # Example 3: Larger numbers
        print("Example 3: Larger numbers (23 + 45 = 68)")
        self.visualize_addition_on_helix(23, 45, period=100)
        
        # Example 4: Animation of addition
        print("Example 4: Animation of the Clock algorithm (8 + 6 = 14)")
        animation = self.create_animation_of_addition(8, 6, period=10)
        
        # Example 5: Helix projection
        print("Example 5: Helical representation projected to a circle")
        self.visualize_helix_projection(period=10)

if __name__ == "__main__":
    # Create visualization instance
    viz = ClockAlgorithmVisualization()
    
    # Demonstrate the helix representation
    print("Visualizing number representation on a helix...")
    viz.plot_number_helix(num_range=(0, 50), period=10)
    
    # Demonstrate the Clock algorithm
    print("Demonstrating the Clock algorithm for addition...")
    viz.demonstrate_clock_algorithm() 