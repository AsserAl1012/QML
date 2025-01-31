import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Number of qubits = number of features (for simplicity)
n_qubits = 5  # Adjust based on your data
dev = qml.device("lightning.qubit", wires=n_qubits)

# Sample input and weights for demonstration
inputs = np.random.randn(n_qubits)  # Random example input
weights = np.random.randn(4 * n_qubits)  # 2 layers of trainable weights

# Define the parameterized quantum circuit
@qml.qnode(dev)
def qnn_circuit(inputs, weights):
    # Step 1: Encode input features using rotation gates
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Step 2: First layer of parameterized rotations
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
        qml.RZ(weights[n_qubits + i], wires=i)
    
    # First layer of entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Step 3: Second layer of parameterized rotations
    for i in range(n_qubits):
        qml.RY(weights[2 * n_qubits + i], wires=i)
        qml.RZ(weights[3 * n_qubits + i], wires=i)
    
    # Second layer of entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Draw the circuit using PennyLane's drawer
qml.draw_mpl(qnn_circuit, decimals=2)(inputs, weights)
plt.show()
