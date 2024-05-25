# OCNN - Oscillator Chaotic Neural Network

OCNN is a Python library for data clustering using oscillator chaotic neural networks (OCNN). It provides tools for creating and visualizing these networks, and for analyzing their state using mutual information.

## Features
- **OCNN Construction**: Build oscillator chaotic neural networks with customizable interaction and activation functions.
- **Visualization**: Plot network states, hierarchical clustering, and clusters.
- **Analysis**: Compute mutual information and perform clustering based on linkage and mutual information.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
### OCNN Initialization
Initialize the OCNN with your data and specified parameters:
```python
from ocnn import OCNN

input_data = ...  # your input data as a numpy array
ocnn_model = OCNN(input_data)
```

### Running the Network
Run the network to obtain the states:
```python
net_states = ocnn_model(observation_count=100, transfer_count=10, seed=42)
```

### Visualization
Use `OCNNetViz` to visualize the network states and clusters:
```python
from ocnn import OCNNetViz

# Plot network states
OCNNetViz.plot_states(net_states)

# Plot hierarchical clustering
OCNNetViz.plot_hierarchy(net_states)

# Plot clusters
input_data = ...  # your input data as a numpy array
OCNNetViz.plot_clusters(input_data, net_states, theta=0.5)
```

### Analysis
```python
from ocnn import OCNNetAnalyser

# Calculate mutual information
mutual_info = OCNNetAnalyser.calc_mutual_info(net_states)

# Perform clustering
clusters = OCNNetAnalyser.cluster(net_states, theta=0.5)
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.