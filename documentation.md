# BNAI Project Documentation

## Project Overview
The BNAI (Behavioral Neural AI) project is a system that analyzes and generates behavioral parameters for AI models. It employs a neural network architecture to learn and generate BNAI profiles that characterize AI model behavior.

## Project Structure
```
src/
  ├── data/
  │   ├── io.py            # Data loading and saving utilities
  │   ├── preprocess.py     # Data preprocessing functions
  │   └── synthetic_data.py # Synthetic data generation
  ├── losses/
  │   └── bnai_loss.py     # Custom loss functions
  ├── main.py              # Main entry point
  ├── model/
  │   ├── bnai_model.py    # BNAI model architecture
  │   ├── layers.py        # Custom neural network layers
  │   └── training.py      # Training loop implementation
  └── utils/
      ├── bnai_calculator.py # BNAI parameter calculation
      ├── config.py         # Configuration handling
      ├── config.yaml       # Configuration settings
      └── logger.py         # Logging utilities
```

## Key Components

### 1. BNAI Formula and Parameters

#### BNAI Score Formula
The optimized formula for calculating the BNAI score is:

\[
\boxed{
\begin{aligned}
\text{BNAI} = \frac{ 
\displaystyle \sum_{i \in \{A,\, E_g,\, G,\, H,\, I,\, L,\, P,\, Q,\, R,\, U,\, V,\, O,\, Etica,\, Sicurezza\}} w_i \cdot \ln(i + 1) \,\, + \,\, \sigma(R_{TL}) \cdot \sigma(R_{ML}) \cdot e^{-\lambda_{reg}}
}{ 
\displaystyle 1 + \alpha \cdot (B + C + E) + \beta \cdot e^{t/M}
}
\end{aligned}
\]

where:
- \(w_i\) are predefined weights for each parameter (their sum is normalized to 1).
- \(\sigma(x) = \dfrac{2}{1 + e^{-x}}\) normalizes \(R_{TL}\) and \(R_{ML}\) in the range [0,2].
- \(\alpha\) and \(\beta\) are balancing coefficients (default: \(\alpha = 0.5\), \(\beta = 0.1\)).

##### Parameter Unification
1. **Decomposition and Unification:**
   - The parameter **\(G_c\)** (Growth Factor) is integrated into \(G\) through the temporal derivative (\(\Delta G/\Delta t\)).
   - **\(T\)** (Tensor Dimensionality) is absorbed into **\(C\)** (Model Complexity).

#### Parameter Definitions and Measurements

The system calculates and manages the following BNAI parameters:

##### Core Parameters
- **A (Adaptability)**: Range [0.5, 2.0]
  - Measures ability to adapt to domain variations
  - Calculated as performance variation on out-of-distribution datasets

- **E_g (Generational Evolution)**: Range [0.8, 1.5]
  - Average performance improvement between versions
  - Measured through standard benchmark comparisons

- **G (Growth)**: Range [1.0, 6.0]
  - Computational capacity based on parameter count
  - Calculated as log₁₀(number of parameters)

- **G_c (Growth Factor)**: Range [0.0, 2.0]
  - Complexity increase rate over time
  - Measured as parameter count variation percentage

- **H (Learning Entropy)**: Range [0.0, 5.0]
  - Information acquisition measurement
  - Calculated as input-output entropy differential

- **I (Interconnection)**: Range [0.0, 2.0]
  - Knowledge integration capability
  - Measured through transfer learning performance differential

- **L (Learning Level)**: Range [0.0, 1.0]
  - Model convergence state
  - Based on epochs needed for loss stabilization

- **P (Computational Power)**: Range [1.0, 1000.0]
  - Resource utilization measurement
  - Based on FLOPS and FLOPS/performance ratio

- **Q (Response Precision)**: Range [0.0, 1.0]
  - Model accuracy measurement
  - Using standard metrics (accuracy, F1-score)

- **R (Robustness)**: Range [0.0, 2.0]
  - Stability under perturbations
  - Performance variation under attacks/noise

- **S (Current State)**: Range [0.0, 1.0]
  - Aggregate performance evaluation
  - Normalized benchmark scores

- **U (Autonomy)**: Range [0.0, 1.0]
  - Operational independence
  - Ratio of local to delegated operations

- **V (Response Speed)**: Range [0.001, 10.0]
  - Average inference time
  - Measured in milliseconds/seconds

- **T (Tensor Dimensionality)**: Range [1.0, 100.0]
  - Architectural complexity
  - Normalized average tensor dimensions

- **O (Computational Efficiency)**: Range [0.0, 1.0]
  - Performance/cost ratio
  - Accuracy divided by computational cost

##### Penalty Parameters
- **B (Bias)**: Range [0.0, 1.0]
  - Performance disparity across data subsets
  - Measured using fairness metrics

- **C (Complexity)**: Range [0.0, 100.0]
  - Overall architectural complexity
  - Based on total operations count

- **E (Learning Error)**: Range [0.0, 1.0]
  - Average training errors
  - Normalized loss value

##### Advanced Parameters
- **M (Memory Capacity)**: Range [1.0, 100.0]
  - Information storage capability
  - Based on state vector dimensions

- **R_TL (Transfer Learning Coefficient)**: Range [0.5, 2.0]
  - Knowledge transfer benefit measurement
  - Calculated as: 1 + (Performance_TL - Performance_base)/Performance_base

- **R_ML (Meta-Learning Coefficient)**: Range [0.5, 2.0]
  - Rapid task adaptation capability
  - Based on meta-dataset performance

### 2. Model Architecture
The BNAI model uses a latent space of dimension 100 and generates 21 BNAI parameters through a neural network with:
- Hidden layers: [256, 512]
- Activation function: ReLU
- Output dimension: 21

### 3. Training Configuration
- Learning rate: 0.001
- Batch size: 32
- Training epochs: 100
- Validation split: 0.2
- L2 regularization with alpha: 0.01

### 4. Data Processing
- Supports synthetic data generation
- Implements data normalization
- Provides train/validation split functionality
- Includes data augmentation options (currently disabled)

### 5. Key Implementations

#### BNAI Parameter Calculation
The `calculate_bnai_parameters` function in `bnai_calculator.py` computes BNAI parameters for a given model:
- Calculates model complexity using parameter count
- Computes tensor dimensionality
- Evaluates regularization terms
- Assesses model characteristics like adaptability and robustness

#### Training Loop
The training implementation in `training.py` includes:
- Adam optimizer
- Custom BNAI loss function
- Validation monitoring
- Model checkpointing
- GPU support

#### Data Management
The data pipeline supports:
- JSON file I/O
- Data preprocessing and normalization
- Synthetic data generation with configurable parameters
- Train/validation data splitting

## Usage
The system is configured through `config.yaml` and can be run using:
```bash
python src/main.py --config src/utils/config.yaml --epochs 100 --data_path <path_to_data> --save_path models/bnai_clone.pth
```

## Future Improvements
1. Implementation of placeholder functions in `bnai_calculator.py`
2. Enhanced data augmentation strategies
3. Additional model architectures
4. Improved parameter calculation methods
5. Extended validation metrics
6. Integration of interpretability and explainability metrics
7. Implementation of calibration measurements
8. Enhanced security testing capabilities
9. Integration of ethical consideration metrics
10. Task-specific metric implementations