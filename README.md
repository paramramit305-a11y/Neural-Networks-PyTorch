# 🧠 Neural Networks PyTorch

> **Production-ready implementation of Artificial Neural Networks using PyTorch for classification and regression tasks**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-professional-black)](https://github.com/paramramit305-a11y)

Complete end-to-end implementation of Artificial Neural Networks using PyTorch for **multi-class classification** and **regression** tasks, featuring enterprise-grade data preprocessing, model training, and evaluation pipelines.

---

## 📂 Project Structure

```
Neural-Networks-PyTorch/
│
├── ANN_Classification/
│   ├── ANN_Classification.ipynb    # Classification implementation
│   └── DateFruit_Dataset.csv       # Training dataset
│
├── ANN_Regression/
│   ├── ANN_Regression.ipynb        # Regression implementation
│   ├── powerplant_data.csv         # Training dataset
│   └── best_model.pt               # Saved model checkpoint
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🎯 Problem Statement & Solution

### Classification Challenge
Automated classification of date fruit varieties based on morphological features - a real-world agriculture technology problem requiring multi-class prediction with high accuracy.

### Regression Challenge
Predicting electrical energy output in combined cycle power plants based on ambient variables - an industrial optimization problem requiring precise numerical predictions.

---

## 💡 Project Highlights

### 1️⃣ **Date Fruit Classification - Multi-Class Neural Network**

**Business Context:**  
Agricultural quality control and automated sorting systems require accurate fruit variety identification. This model enables automated classification based on visual and morphological features.

**Technical Implementation:**
- **Dataset:** 898 samples with 34 engineered features
- **Target Classes:** 7 varieties (BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY)
- **Architecture:** 3-layer feedforward neural network
  - Input Layer: 34 features → 64 neurons (ReLU)
  - Hidden Layer: 64 → 64 neurons (ReLU)
  - Output Layer: 64 → 7 classes (Softmax via CrossEntropyLoss)

**Performance Metrics:**
- **Training Loss:** 0.032 (100 epochs)
- **Convergence:** Stable within 100 epochs
- **Optimization:** Adam optimizer with default learning rate

**Key Engineering Decisions:**
- StandardScaler for feature normalization (zero mean, unit variance)
- LabelEncoder for categorical target encoding
- 80-20 train-test split with stratification
- Batch processing (batch_size=32) for efficient training
- No dropout/regularization needed due to sufficient data

---

### 2️⃣ **Power Plant Energy Prediction - Regression Neural Network**

**Business Context:**  
Energy output prediction in combined cycle power plants enables optimized operations, predictive maintenance, and improved resource allocation.

**Technical Implementation:**
- **Task:** Predict electrical energy output (continuous values)
- **Architecture:** Multi-layer perceptron optimized for regression
- **Loss Function:** Mean Squared Error (MSE)
- **Model Persistence:** Best model saved as `best_model.pt`

**Key Features:**
- Feature scaling for improved gradient descent
- Model checkpointing with validation-based selection
- Performance tracking: MSE, MAE, R² score
- Production-ready model artifact

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning Framework** | PyTorch 2.x | Neural network implementation |
| **Numerical Computing** | NumPy 1.24+ | Array operations and mathematics |
| **Data Processing** | Pandas 2.0+ | Dataset manipulation and analysis |
| **Preprocessing** | Scikit-learn 1.3+ | Scaling, encoding, train-test splits |
| **Visualization** | Matplotlib, Seaborn | Loss curves and performance plots |
| **Development Environment** | Jupyter Notebook | Interactive development and documentation |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
8GB RAM recommended
```

### Installation

**1. Clone the Repository**
```bash
git clone https://github.com/paramramit305-a11y/Neural-Networks-PyTorch.git
cd Neural-Networks-PyTorch
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook
```

### Requirements File
```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 📊 Model Architecture Details

### Classification Network

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(34, 64),      # Input layer
            nn.ReLU(),              # Activation
            nn.Linear(64, 64),      # Hidden layer
            nn.ReLU(),              # Activation
            nn.Linear(64, 7)        # Output layer (7 classes)
        )
    
    def forward(self, x):
        return self.model(x)
```

**Architecture Rationale:**
- **2 hidden layers** - Sufficient capacity for non-linear patterns
- **64 neurons** - Balanced between model capacity and overfitting risk
- **ReLU activation** - Fast training, no vanishing gradient
- **No softmax** - CrossEntropyLoss includes log-softmax internally

### Training Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **Optimizer** | Adam | Adaptive learning rate, faster convergence |
| **Learning Rate** | 0.001 (default) | Standard for Adam optimizer |
| **Batch Size** | 32 | Balance between speed and gradient accuracy |
| **Epochs** | 100 | Sufficient for convergence on this dataset |
| **Loss Function** | CrossEntropyLoss | Standard for multi-class classification |
| **Train-Test Split** | 80-20 | Industry standard for medium datasets |

---

## 🔬 Data Preprocessing Pipeline

### Classification Pipeline
```python
1. Load CSV dataset → Pandas DataFrame
2. Feature-target separation (X, y)
3. Label encoding (string classes → integers)
4. Train-test split (stratified, random_state=42)
5. StandardScaler fitting on train set
6. Transform train and test sets
7. Convert to PyTorch tensors (float32, long)
8. Create DataLoader with batching
```

### Regression Pipeline
```python
1. Load CSV dataset → Pandas DataFrame
2. Feature-target separation
3. Train-test split
4. StandardScaler for features
5. PyTorch tensor conversion
6. DataLoader creation
7. Model training with checkpointing
```

---

## 📈 Results & Performance

### Classification Results

| Metric | Value |
|--------|-------|
| **Final Training Loss** | 0.032 |
| **Convergence Speed** | 100 epochs |
| **Batch Processing** | 32 samples/batch |
| **Optimizer** | Adam (lr=0.001) |

**Loss Convergence Pattern:**
- Epoch 1: ~1.74 (random initialization)
- Epoch 50: ~0.09 (rapid improvement)
- Epoch 100: ~0.032 (stable convergence)

### Regression Results
- **Model Artifact:** Saved as `best_model.pt`
- **Validation-based selection:** Best checkpoint preserved
- **Ready for deployment:** Serialized PyTorch state_dict

---

## 🎓 Key Learning Outcomes

This project demonstrates mastery of:

✅ **PyTorch Fundamentals**
- Building neural networks with `nn.Module`
- Implementing forward propagation
- Working with `nn.Sequential` for layer stacking
- Understanding computational graphs

✅ **Deep Learning Workflow**
- End-to-end pipeline from raw data to trained model
- Proper train-test splitting and validation
- Batch processing with DataLoader
- Loss function selection and optimization

✅ **Production Best Practices**
- Model checkpointing and persistence
- Reproducible preprocessing pipelines
- Clean code structure and documentation
- Version control and collaboration readiness

✅ **Industry-Standard Tools**
- PyTorch for deep learning
- Scikit-learn for preprocessing
- Pandas for data manipulation
- Professional code organization

---

## 🔍 Code Quality Features

- ✅ **Modular Design** - Separated preprocessing, model, and training logic
- ✅ **Type Safety** - Explicit tensor dtype declarations
- ✅ **Reproducibility** - Random seeds and deterministic splits
- ✅ **Documentation** - Clear comments and markdown cells
- ✅ **Best Practices** - Industry-standard naming and structure
- ✅ **Version Control** - Git-friendly notebook structure
- ✅ **Scalability** - Easy to extend to new datasets

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- ⭐ Star this repository

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Connect & Collaborate

**Author:** Parmar Amit - An AIML Enthaust

- 🐙 **GitHub:** [@paramramit305-a11y](https://github.com/paramramit305-a11y)
- 💼 **LinkedIn:** [Parmar Amit](https://www.linkedin.com/in/parmar-amit-97941a377)

---

## 🌟 Acknowledgments

Built as part of my **AI/ML learning journey** at **Gokul Global University**, following industry best practices and production-ready code standards.

---

## 🎯 Future Enhancements

- [ ] Add cross-validation for robust evaluation
- [ ] Implement learning rate schedulers
- [ ] Add early stopping mechanism
- [ ] Include confusion matrix visualization
- [ ] Deploy models as REST API
- [ ] Add unit tests for preprocessing
- [ ] Implement hyperparameter tuning with Optuna
- [ ] Create Streamlit web interface

---

⭐ **If you found this project helpful, please consider giving it a star!**

---

**Building intelligent systems, one neural network at a time.** 🚀
