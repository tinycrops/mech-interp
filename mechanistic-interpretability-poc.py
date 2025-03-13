import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import random
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MechanisticInterpretabilityEnvironment:
    def __init__(self):
        # Load and preprocess MNIST data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Create a simple model
        self.model = self.create_model()
        
        # Create a functional version of the model for layer activation extraction
        self.functional_model = self.create_functional_model()
        
        # Train the model on original MNIST
        self.train_model()
        
        # Create "updated API version" by modifying some digits
        self.create_modified_dataset()
        
        # Track performance metrics
        self.metrics = {
            'original_accuracy': [],
            'modified_accuracy': [],
            'weight_changes': []
        }
        
        self.current_layer_index = 0
        self.current_weight_indices = (0, 0)
        self.activation_cache = {}
        self.weight_change_history = []
        
    def create_model(self):
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu', name='dense_1'),
            Dense(64, activation='relu', name='dense_2'),
            Dense(10, activation='softmax', name='output')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def create_functional_model(self):
        """Create a functional version of the model for layer activation extraction"""
        # Define the input layer
        inputs = Input(shape=(28, 28))
        
        # Recreate the model architecture using the functional API
        x = Flatten()(inputs)
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dense(64, activation='relu', name='dense_2')(x)
        outputs = Dense(10, activation='softmax', name='output')(x)
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def train_model(self):
        self.model.fit(self.x_train, self.y_train, 
                      batch_size=128, 
                      epochs=3, 
                      verbose=1,
                      validation_data=(self.x_test, self.y_test))
        
        # Copy weights from sequential model to functional model
        for i, layer in enumerate(self.model.layers):
            if i > 0:  # Skip the Flatten layer (index 0)
                self.functional_model.get_layer(layer.name).set_weights(layer.get_weights())
        
    def create_modified_dataset(self):
        """Create a modified version of the dataset to simulate API changes"""
        # Create a copy of the test set
        self.x_modified = self.x_test.copy()
        self.y_modified = self.y_test.copy()
        
        # Select digit to modify (let's say we modify the appearance of digit 3)
        target_digit = 3
        digit_indices = np.where(self.y_test == target_digit)[0]
        
        # Apply a transformation to all instances of digit 3
        # (e.g., add a horizontal line in the middle)
        for idx in digit_indices:
            # Add a horizontal line in the middle (row 14)
            self.x_modified[idx, 14, 10:18] = 1.0
        
        print(f"Modified {len(digit_indices)} instances of digit {target_digit}")
        
        # Calculate baseline performance on modified dataset
        _, self.baseline_modified_acc = self.model.evaluate(
            self.x_modified, self.y_modified, verbose=0
        )
        print(f"Baseline accuracy on modified dataset: {self.baseline_modified_acc:.4f}")
    
    def get_layer_weights(self, layer_index):
        """Get the weights of a specific layer"""
        return self.model.layers[layer_index].get_weights()[0]
    
    def set_layer_weights(self, layer_index, new_weights):
        """Update weights for a specific layer"""
        weights = self.model.layers[layer_index].get_weights()
        weights[0] = new_weights
        self.model.layers[layer_index].set_weights(weights)
    
    def compute_neuron_activations(self, input_image, layer_name):
        """Compute activations for a specific layer given an input"""
        # Use the functional model to extract activations
        layer_model = Model(inputs=self.functional_model.input, 
                           outputs=self.functional_model.get_layer(layer_name).output)
        return layer_model.predict(np.expand_dims(input_image, axis=0))[0]
    
    def find_important_weights(self, error_examples, layer_index, top_n=5):
        """Identify weights that might be important for fixing errors"""
        layer = self.model.layers[layer_index]
        layer_name = layer.name
        
        # Get layer weights
        weights_list = layer.get_weights()
        if not weights_list:
            print(f"Layer {layer_name} has no weights")
            return []
            
        weights = weights_list[0]
        weight_importance = np.zeros_like(weights)
        
        # Compute activations for each error example
        for img, true_label in error_examples:
            # Get activations of the previous layer
            if layer_index == 1:  # First dense layer after flatten
                prev_activations = img.flatten()
            else:
                prev_layer = self.model.layers[layer_index-1].name
                prev_activations = self.compute_neuron_activations(img, prev_layer)
            
            # Get activations of this layer
            curr_activations = self.compute_neuron_activations(img, layer_name)
            
            # Simple importance metric: product of input activation and output activation
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    weight_importance[i, j] += abs(prev_activations[i] * curr_activations[j])
        
        # Flatten and find top N important weights
        flat_importance = weight_importance.flatten()
        top_indices = np.argsort(flat_importance)[-top_n:]
        
        # Convert flat indices back to 2D indices
        top_weights = []
        for idx in top_indices:
            i, j = np.unravel_index(idx, weights.shape)
            top_weights.append((i, j, weights[i, j], flat_importance[idx]))
        
        return top_weights
    
    def modify_weight(self, layer_index, weight_indices, delta):
        """Modify a specific weight and evaluate the impact"""
        # Get current weights
        weights = self.get_layer_weights(layer_index)
        i, j = weight_indices
        
        # Store original weight value
        original_value = weights[i, j]
        
        # Modify the weight
        weights[i, j] += delta
        self.set_layer_weights(layer_index, weights)
        
        # Evaluate on both datasets
        _, orig_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        _, mod_acc = self.model.evaluate(self.x_modified, self.y_modified, verbose=0)
        
        # Record metrics
        self.metrics['original_accuracy'].append(orig_acc)
        self.metrics['modified_accuracy'].append(mod_acc)
        self.metrics['weight_changes'].append({
            'layer': layer_index,
            'i': i,
            'j': j,
            'before': float(original_value),
            'after': float(weights[i, j]),
            'delta': float(delta)
        })
        
        # Keep track of weight change history
        self.weight_change_history.append({
            'layer': layer_index,
            'weight_indices': (i, j),
            'original_value': float(original_value),
            'new_value': float(weights[i, j]),
            'original_accuracy': float(orig_acc),
            'modified_accuracy': float(mod_acc)
        })
        
        return {
            'original_accuracy': orig_acc,
            'modified_accuracy': mod_acc,
            'change': delta
        }
    
    def reset_weight(self, history_index):
        """Reset a weight to its original value"""
        if 0 <= history_index < len(self.weight_change_history):
            change = self.weight_change_history[history_index]
            layer_index = change['layer']
            i, j = change['weight_indices']
            original_value = change['original_value']
            
            # Get current weights
            weights = self.get_layer_weights(layer_index)
            
            # Reset to original value
            weights[i, j] = original_value
            self.set_layer_weights(layer_index, weights)
            
            return True
        return False
    
    def get_error_examples(self, n=10):
        """Get examples where the model fails on the modified dataset"""
        predictions = self.model.predict(self.x_modified)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Find misclassified examples
        errors = np.where(pred_classes != self.y_modified)[0]
        
        # Take a random sample of errors
        if len(errors) > n:
            sample_indices = np.random.choice(errors, size=n, replace=False)
        else:
            sample_indices = errors
        
        return [(self.x_modified[i], self.y_modified[i]) for i in sample_indices]
    
    def visualize_error_example(self, example_index=0):
        """Visualize an example where the model fails"""
        error_examples = self.get_error_examples()
        if not error_examples:
            print("No errors found!")
            return
        
        img, true_label = error_examples[example_index % len(error_examples)]
        pred_label = np.argmax(self.model.predict(np.expand_dims(img, axis=0))[0])
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}, Predicted: {pred_label}")
        plt.colorbar()
        plt.show()
    
    def run_interactive_game(self):
        """Run interactive environment to modify weights"""
        # Get initial error examples
        error_examples = self.get_error_examples()
        if not error_examples:
            print("No errors found! The model is already performing well on the modified dataset.")
            return
        
        # Choose an error example to focus on
        focus_example, true_label = error_examples[0]
        pred_label = np.argmax(self.model.predict(np.expand_dims(focus_example, axis=0))[0])
        
        # Find important weights for the output layer
        important_weights = self.find_important_weights(
            error_examples, layer_index=2, top_n=5
        )
        
        # Setup interactive plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)
        
        # Plot the focus example
        axs[0, 0].imshow(focus_example, cmap='gray')
        axs[0, 0].set_title(f"Error Example\nTrue: {true_label}, Pred: {pred_label}")
        
        # Plot original vs modified digit
        original_3 = self.x_test[np.where(self.y_test == 3)[0][0]]
        modified_3 = self.x_modified[np.where(self.y_modified == 3)[0][0]]
        
        axs[0, 1].imshow(original_3, cmap='gray')
        axs[0, 1].set_title("Original Digit 3")
        
        axs[1, 0].imshow(modified_3, cmap='gray')
        axs[1, 0].set_title("Modified Digit 3")
        
        # Placeholder for performance plot
        accuracy_ax = axs[1, 1]
        accuracy_ax.set_title("Accuracy Metrics")
        accuracy_ax.set_xlabel("Weight Modification Steps")
        accuracy_ax.set_ylabel("Accuracy")
        orig_line, = accuracy_ax.plot(self.metrics['original_accuracy'] + [self.baseline_modified_acc], 
                                     'b-', label='Original Data')
        mod_line, = accuracy_ax.plot(self.metrics['modified_accuracy'] + [self.baseline_modified_acc], 
                                    'r-', label='Modified Data')
        accuracy_ax.legend()
        accuracy_ax.set_ylim(0, 1)
        
        # Add weight slider
        ax_weight = plt.axes([0.25, 0.15, 0.65, 0.03])
        weight_slider = Slider(
            ax=ax_weight,
            label='Weight Adjustment',
            valmin=-0.5,
            valmax=0.5,
            valinit=0.0,
            valstep=0.01
        )
        
        # Add layer and neuron selection
        ax_layer = plt.axes([0.25, 0.1, 0.65, 0.03])
        layer_slider = Slider(
            ax=ax_layer,
            label='Layer (0=dense_1, 1=dense_2, 2=output)',
            valmin=0,
            valmax=2,
            valinit=2,
            valstep=1
        )
        
        # Text display for current weight info
        weight_info_text = plt.figtext(0.5, 0.05, "Select weight to modify", ha="center")
        
        # Current state
        current_state = {
            'layer_index': 2,
            'weight_i': important_weights[0][0],
            'weight_j': important_weights[0][1],
            'weight_value': important_weights[0][2],
            'delta': 0.0
        }
        
        def update_weight_info():
            layer_index = int(current_state['layer_index'])
            i, j = current_state['weight_i'], current_state['weight_j']
            weights = self.get_layer_weights(layer_index)
            current_value = weights[i, j]
            layer_name = self.model.layers[layer_index].name
            
            weight_info_text.set_text(
                f"Layer: {layer_name}, Weight[{i},{j}] = {current_value:.6f}, " 
                f"Delta: {current_state['delta']:.6f}"
            )
        
        def on_layer_change(val):
            current_state['layer_index'] = int(val)
            layer_index = current_state['layer_index']
            
            # Find important weights for this layer
            new_important_weights = self.find_important_weights(
                error_examples, layer_index=layer_index, top_n=5
            )
            
            if new_important_weights:
                current_state['weight_i'] = new_important_weights[0][0]
                current_state['weight_j'] = new_important_weights[0][1]
                current_state['weight_value'] = new_important_weights[0][2]
            else:
                # If no important weights found, try the next layer
                if layer_index < 2:
                    layer_slider.set_val(layer_index + 1)
                else:
                    print(f"No weights found for layer {layer_index}. Please select another layer.")
                    # Set to a default value to avoid errors
                    current_state['weight_i'] = 0
                    current_state['weight_j'] = 0
                    current_state['weight_value'] = 0.0
            
            update_weight_info()
            fig.canvas.draw_idle()
        
        def on_weight_change(val):
            current_state['delta'] = val
            update_weight_info()
        
        def apply_weight_change(event):
            layer_index = current_state['layer_index']
            i, j = current_state['weight_i'], current_state['weight_j']
            delta = current_state['delta']
            
            result = self.modify_weight(layer_index, (i, j), delta)
            
            # Update the plot
            orig_line.set_data(range(len(self.metrics['original_accuracy'])), 
                              self.metrics['original_accuracy'])
            mod_line.set_data(range(len(self.metrics['modified_accuracy'])), 
                             self.metrics['modified_accuracy'])
            accuracy_ax.set_xlim(0, max(1, len(self.metrics['original_accuracy'])-1))
            
            # Reset slider
            weight_slider.set_val(0.0)
            current_state['delta'] = 0.0
            
            # Update weight info
            update_weight_info()
            
            # Redraw
            fig.canvas.draw_idle()
            
            # Update prediction for focus example
            new_pred = np.argmax(self.model.predict(np.expand_dims(focus_example, axis=0))[0])
            axs[0, 0].set_title(f"Error Example\nTrue: {true_label}, Pred: {new_pred}")
            
            print(f"Applied weight change: {delta} to layer {layer_index} weight [{i},{j}]")
            print(f"Original accuracy: {result['original_accuracy']:.4f}, "
                  f"Modified accuracy: {result['modified_accuracy']:.4f}")
        
        # Create button for applying weight changes
        ax_button = plt.axes([0.8, 0.01, 0.15, 0.04])
        apply_button = Button(ax_button, 'Apply Change')
        apply_button.on_clicked(apply_weight_change)
        
        # Connect sliders
        layer_slider.on_changed(on_layer_change)
        weight_slider.on_changed(on_weight_change)
        
        # Initialize weight info
        update_weight_info()
        
        # Show plot
        plt.show()
        
    def export_results(self, filename='weight_modifications.json'):
        """Export the results of weight modifications"""
        # Convert NumPy types to Python native types
        def convert_to_native_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native_types(item) for item in obj)
            else:
                return obj
        
        results = {
            'metrics': self.metrics,
            'weight_change_history': self.weight_change_history,
            'baseline_modified_accuracy': float(self.baseline_modified_acc)
        }
        
        # Convert all NumPy types to native Python types
        results = convert_to_native_types(results)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")

# Running the environment
if __name__ == "__main__":
    env = MechanisticInterpretabilityEnvironment()
    
    # Print initial accuracies
    _, orig_acc = env.model.evaluate(env.x_test, env.y_test, verbose=0)
    _, mod_acc = env.model.evaluate(env.x_modified, env.y_modified, verbose=0)
    
    print(f"Initial accuracy on original test set: {orig_acc:.4f}")
    print(f"Initial accuracy on modified test set: {mod_acc:.4f}")
    
    # Run the interactive game
    env.run_interactive_game()
    
    # Export results
    env.export_results()
    
    # Alternative: run an automated search for weight adjustments
    # For a more automated approach, uncomment the following code
    """
    def automated_weight_search():
        # Get error examples
        error_examples = env.get_error_examples(n=5)
        
        # Try modifying weights in the output layer
        for layer_idx in [2, 1]:
            important_weights = env.find_important_weights(
                error_examples, layer_index=layer_idx, top_n=10
            )
            
            for i, j, weight_val, importance in important_weights:
                # Try small positive adjustment
                result_pos = env.modify_weight(layer_idx, (i, j), 0.1)
                
                # If accuracy improves, keep it; otherwise try negative
                if result_pos['modified_accuracy'] <= mod_acc:
                    # Reset and try negative
                    env.reset_weight(len(env.weight_change_history) - 1)
                    result_neg = env.modify_weight(layer_idx, (i, j), -0.1)
                    
                    # If still no improvement, reset
                    if result_neg['modified_accuracy'] <= mod_acc:
                        env.reset_weight(len(env.weight_change_history) - 1)
        
        # Print final accuracies
        _, final_orig_acc = env.model.evaluate(env.x_test, env.y_test, verbose=0)
        _, final_mod_acc = env.model.evaluate(env.x_modified, env.y_modified, verbose=0)
        
        print(f"Final accuracy on original test set: {final_orig_acc:.4f}")
        print(f"Final accuracy on modified test set: {final_mod_acc:.4f}")
        print(f"Improvement on modified test set: {final_mod_acc - mod_acc:.4f}")
    
    # automated_weight_search()
    """
