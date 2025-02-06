import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

class TransformerActivationAnalyzer:
    def __init__(self, model_name="bert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def get_activations(self, text, layer=-1):
        """Extract activations from a specified transformer layer."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer].squeeze(0)  # Shape: (seq_len, hidden_dim)
    
    def get_neuron_importance(self, text, layer=-1):
        """Compute the importance of neurons in a given layer based on activation magnitude."""
        activations = self.get_activations(text, layer)
        return activations.abs().mean(dim=0)  # Average importance across tokens
    
    def plot_neuron_importance(self, text, layer=-1):
        """Visualize neuron importance in a given layer."""
        importance = self.get_neuron_importance(text, layer)
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(importance)), importance.numpy())
        plt.xlabel("Neuron Index")
        plt.ylabel("Importance Score")
        plt.title(f"Neuron Importance in Layer {layer}")
        plt.show()

if __name__ == "__main__":
    analyzer = TransformerActivationAnalyzer("bert-base-uncased")
    sample_text = "Anthropic is advancing AI interpretability."
    analyzer.plot_neuron_importance(sample_text, layer=-1)
