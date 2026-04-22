# 🧠 Neural Networks PyTorch

Complete implementation of Artificial Neural Networks using PyTorch for **classification** and **regression** tasks, featuring end-to-end data preprocessing, model training, and evaluation pipelines.

## 📂 Project Structure

```
Neural-Networks-PyTorch/
│
├── Classification/
│   ├── ANN_Classification.ipynb
│   └── DateFruit_Dataset.csv
│
├── Regression/
│   ├── ANN_Regression.ipynb
│   ├── powerplant_data.csv
│   └── best_model.pt
│
├── README.md
└── requirements.txt
```

---

## 🎯 Projects Overview

### 1️⃣ Date Fruit Classification
**Multi-class classification of date fruit varieties using feedforward neural networks**

- **Dataset:** 898 samples | 34 features | 7 classes
- **Classes:** BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY
- **Architecture:** 
  - Input Layer: 34 features
  - Hidden Layers: 2 × 64 neurons with ReLU
  - Output Layer: 7 classes with CrossEntropyLoss
- **Performance:** Training loss ~0.032 in 100 epochs
- **Techniques:**
  - StandardScaler for feature normalization
  - LabelEncoder for target encoding
  - Train-test split (80-20)
  - Batch processing (batch_size=32)

**Key Highlights:**
- Complete data preprocessing pipeline
- Adam optimizer with default learning rate
- Loss convergence visualization
- Model evaluation on test set

---

### 2️⃣ Power Plant Energy Output Prediction
**Regression model to predict electrical energy output based on plant parameters**

- **Dataset:** Power plant operational data
- **Task:** Predict energy output (regression)
- **Architecture:** Multi-layer perceptron with MSE loss
- **Model Artifact:** Saved as `best_model.pt`
- **Techniques:**
  - Feature scaling for better convergence
  - Model checkpointing
  - Performance metrics: MSE, MAE

**Key Highlights:**
- End-to-end regression pipeline
- Model persistence with PyTorch
- Hyperparameter tuning
- Validation-based early stopping

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch 2.x |
| **Data Processing** | Pandas, NumPy |
| **Preprocessing** | Scikit-learn (StandardScaler, LabelEncoder) |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook |

---

## 📦 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/paramramit305-a11y/Neural-Networks-PyTorch.git
cd Neural-Networks-PyTorch

# Install required packages
pip install -r requirements.txt
```

### requirements.txt
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

## 🚀 Usage

### Classification Task
```bash
cd Classification
jupyter notebook ANN_Classification.ipynb
```

**Workflow:**
1. Load and explore DateFruit dataset
2. Preprocess features with StandardScaler
3. Encode target labels
4. Build 3-layer ANN with PyTorch
5. Train with Adam optimizer
6. Evaluate on test set

### Regression Task
```bash
cd Regression
jupyter notebook ANN_Regression.ipynb
```

**Workflow:**
1. Load power plant dataset
2. Scale features
3. Build regression ANN
4. Train and validate
5. Save best model checkpoint

---

## 📊 Model Architecture

### Classification ANN
```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
```

### Training Configuration
- **Loss Function:** CrossEntropyLoss (Classification), MSELoss (Regression)
- **Optimizer:** Adam
- **Batch Size:** 32
- **Epochs:** 100
- **Train-Test Split:** 80-20

---

## 📈 Results

### Classification Performance
| Metric | Value |
|--------|-------|
| **Final Training Loss** | ~0.032 |
| **Convergence** | 100 epochs |
| **Optimizer** | Adam (lr=0.001) |

### Regression Performance
- Model saved with best validation metrics
- Checkpoint: `best_model.pt`

---

## 🔍 Key Features

✅ Complete data preprocessing pipelines  
✅ PyTorch `nn.Module` implementation  
✅ Batch processing with DataLoader  
✅ Model checkpointing and persistence  
✅ Train-test validation splits  
✅ Loss tracking and visualization  
✅ Industry-standard coding practices  

---

## 📚 Learning Outcomes

This project demonstrates:
- Building ANNs from scratch with PyTorch
- Handling classification and regression tasks
- Feature engineering and scaling
- Model training and evaluation
- Hyperparameter tuning
- Model persistence and deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

---

## 📧 Connect With Me

- **GitHub:** [@paramramit305-a11y](https://github.com/paramramit305-a11y)
- **LinkedIn:** [Aman Banavali](https://www.linkedin.com/in/banavali-aman-97941a377)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🌟 Acknowledgments

Built as part of my **AI/ML learning journey** at Gokul Global University, following industry best practices and production-ready code standards.

---

⭐ **Star this repo if you found it helpful!**

---

**Related Projects:**
- [Credit Risk ML App](https://github.com/paramramit305-a11y/credit-risk-streamlit-app) - Deployed Streamlit app with 88% accuracy
- [KNN Heart Disease Classification](https://github.com/paramramit305-a11y/knn-heart-disease-classification)
- [Employee Turnover Prediction](https://github.com/paramramit305-a11y/Employee-Turnover-Prediction)

### Author - Parmar Amit An AIML Enthaust
