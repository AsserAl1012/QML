# QML
# ### **Dipole Moment Prediction Using Quantum Neural Networks (QNN)**

## **Introduction**  
Molecular dipole moments are essential properties in **quantum chemistry** and **materials science**, influencing **chemical reactivity, solubility, and intermolecular interactions**. Accurate prediction of dipole moments is crucial in areas like **drug discovery, material design, and chemical simulations**.

In this project, we leverage the power of **Quantum Neural Networks (QNNs)** to predict the dipole moments of molecules using data from the **QM9 molecular dataset**. By combining quantum feature encoding with multi-layer parameterized quantum circuits, we aim to showcase the potential of quantum computing for solving complex molecular property prediction tasks.

---

## **Project Overview**  

The project consists of the following key steps:  
1. **Data Parsing and Preprocessing:**  
   - Extract atomic coordinates, charges, and molecular features from **XYZ files** of the QM9 dataset.  
   - Dynamically compute the **dipole moment** using atomic coordinates and charges.  

2. **Quantum Neural Network (QNN) Design:**  
   - Construct a **5-layer parameterized quantum circuit** with rotation gates and entanglement layers to encode molecular features and perform multi-qubit measurements.  

3. **Training and Optimization:**  
   - Train the QNN using **Huber loss** as the objective function, optimized with the **Adam optimizer** and early stopping based on model convergence.  

4. **Evaluation:**  
   - Evaluate the model performance using **Huber loss**, **MSE**, and **R² score**, and compare the predicted dipole moments against actual values.  

---

## **Mathematical Formulation**  

### **Dipole Moment Calculation**  
The dipole moment \( \mathbf{D} \) of a molecule is calculated using the equation:  

$$
\( \mathbf{D} = \sum_{i} q_i \mathbf{r}_i \)
$$

Where:  
- \( q_i \) is the **partial charge** of atom \( i \)  
- \( \mathbf{r}_i \) is the **position vector** of atom \( i \)  

The magnitude of the dipole moment \( D \) is given by:  

$$
D = \| \mathbf{D} \| = \sqrt{D_x^2 + D_y^2 + D_z^2}
$$


To convert the dipole moment from **atomic units (e·bohr)** to **Debye**, we use:  

$$
1 \, \text{e·bohr} = 2.541746 \, \text{Debye}
$$

---

### **Quantum Neural Network Design**  

The QNN is implemented using **PennyLane** and consists of the following layers:  

1. **Input Encoding:**  
   Each feature \( x_i \) is encoded into the quantum state using **rotation gates**:  

$$
\text{RY}(x_i), \quad \text{RX}(x_i), \quad \text{RZ}(x_i)
$$  

2. **Parameterized Layers:**  
   Multiple layers of **parameterized rotations** and **entanglement gates** are applied to the qubits:  

$$
\text{RX}(\theta_i), \quad \text{RY}(\theta_i), \quad \text{RZ}(\theta_i)
$$  

   **Entanglement gates:** We use both **CNOT** and **CZ gates** to create quantum correlations:  

   - **CNOT gate:** Entangles adjacent qubits.  
   - **CZ gate:** Entangles distant qubits for more expressive power.  

3. **Measurement:**  
   The output of the circuit is obtained by measuring the **expectation value** of the **Pauli-Z operator**:  

$$
\langle \sigma_z \rangle = \text{expval}(\text{PauliZ}(0))
$$  

---

## **Huber Loss Function**  

We use **Huber loss** as the objective function, combining the benefits of **mean squared error (MSE)** and **mean absolute error (MAE)**:  

$$
L(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2} (y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\ 
\delta \cdot |y - \hat{y}| - \frac{1}{2} \delta^2 & \text{for } |y - \hat{y}| > \delta  
\end{cases}
$$

Where:  
- \( y \) is the actual dipole moment  
- \( \hat{y} \) is the predicted dipole moment  
- \( \delta \) is a threshold to switch between MSE and MAE  

---

## **Training Pipeline**  

1. **Weight Initialization:**  
   - Initialize weights for the 5-layer QNN:  

$$
\text{weights} \in \mathbb{R}^{6 \times 3 \times n_{\text{qubits}}}
$$  

2. **Optimization Algorithm:**  
   - Use the **Adam optimizer** with a dynamically decaying learning rate:  

$$
\eta_t = \frac{\eta_0}{1 + \text{decay rate} \cdot t}
$$  

3. **Batch Training:**  
   - Train the model using mini-batches of 50 samples to optimize memory usage and improve convergence.  

4. **Early Stopping:**  
   - Stop training if no improvement is seen for 4 consecutive epochs to prevent overfitting.  

---

## **Evaluation Metrics**  

After training, the model is evaluated using:  

1. **Huber Loss:** Combines MSE and MAE for robust evaluation:  

$$
L(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2} (y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\ 
\delta \cdot |y - \hat{y}| - \frac{1}{2} \delta^2 & \text{for } |y - \hat{y}| > \delta  
\end{cases}
$$  

2. **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted dipole moments:  

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$  

3. **Mean Absolute Error (MAE):** Measures the average absolute difference:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$  

4. **R² Score:** Evaluates how well the model explains the variance in the data:  

$$
\R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$  
 
---

## **Technologies and Tools Used**  

- **Quantum Libraries:** PennyLane  
- **Classical ML Libraries:** NumPy, scikit-learn  
- **Data Handling:** Python I/O for XYZ files  
- **Visualization:** Matplotlib  

---

## **Results**  

| Metric      | Value     |  
|-------------|-----------|  
| Huber Loss  | 0.3318    |  
| MSE         | 0.8838    |  
| MAE         | 0.6806    |  
| R² Score    | 0.1213    |  

---

## **Future Improvements**  

- Experiment with **different quantum circuits** for improved expressiveness.  
- Evaluate the performance on additional **molecular properties** like energy gaps and polarizabilities.  
- Explore the use of **quantum kernels** or **quantum feature maps** for classical-quantum hybrid learning.  

---
 
