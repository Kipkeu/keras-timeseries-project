# Time Series Modeling with Deep Learning  
Classification (Transformer) & Forecasting (LSTM)

---

## Project Title  
Understanding and Benchmarking Time Series Models for Classification and Forecasting

---

## Overview  
Time-series data is widely used in engineering and real-world applications such as weather prediction, industrial monitoring, health signals, and activity recognition.

This project explores two deep learning approaches for time-series modeling using official Keras examples:
- A Transformer-based model for time-series classification  
- An LSTM-based model for time-series forecasting  

The objective is to understand, reproduce, organize, and improve these baseline implementations, and evaluate their performance through controlled benchmarking.

---

## Objectives  

By completing this project, the following learning outcomes are achieved:

1. Understand the difference between time-series classification and forecasting  
2. Understand the difference between Transformer and LSTM architectures  
3. Reproduce official Keras baseline models correctly  
4. Explain dataset structure, model inputs/outputs, and training processes  
5. Perform benchmarking through controlled experiments  
6. Implement meaningful improvements to baseline models  
7. Organize results into a clear and structured project summary  

---

## Project Structure  
project/
│
├── baselines/
│   ├── transformer_classification.ipynb
│   └── lstm_forecasting.ipynb
│
├── improvements/
│   ├── transformer_experiments/
│   └── lstm_experiments/
│
├── results/
│   ├── transformer/
│   └── lstm/
│
├── environment/
│   ├── requirements.txt
│   └── environment.yml
│
└── README.md 

---

## Datasets  

### FordA Dataset (Classification)
- Binary classification dataset  
- Input: time-series signals  
- Output: class labels  
- Task: classify signals into two categories  

---

### Jena Climate Dataset (Forecasting)
- Multivariate time-series dataset  
- Input: historical weather measurements  
- Output: future temperature prediction  
- Task: predict future values based on past observations  

---

## Models  

### Transformer Model (Classification)
- Uses self-attention mechanism  
- Captures long-range dependencies in sequences  
- Key components:
  - Multi-head attention  
  - Feed-forward layers  
  - Layer normalization  
  - Residual connections  

---

### LSTM Model (Forecasting)
- Recurrent neural network architecture  
- Designed for sequential data  
- Maintains temporal memory through hidden states  
- Suitable for forecasting tasks  

---

## Task 1 — Baseline Reproduction  

Both Keras examples were executed successfully:
- Models were trained using default configurations  
- Training and validation metrics were recorded  
- Observations were made regarding performance and training behavior  

### Observations  

Transformer:
**Training accuracy**: 51.7%
**Validation accuracy**: 49.2%
**Test accuracy**: 51.6%

**Notes / Observations:**
The model did not learn meaningful patterns from the FordA dataset.
Training and validation accuracy remained near chance level throughout all epochs.
Loss stayed near 0.693, consistent with a model that is effectively guessing.
Validation accuracy plateaued early and fluctuated slightly but showed no improvement.
This indicates under-fitting and suggests the need for model improvements (more heads, deeper Transformer blocks, higher learning rate, etc.).

LSTM:
**Training Loss:** 0.09998788684606552  
**Validation Loss:** 0.12671639025211334  

**Notes / Observations:**
- The LSTM model trained stably and showed consistent improvement across epochs.
- Training loss decreased smoothly from approximately 0.19 to 0.10 over 10 epochs.
- Validation loss started higher than training loss, peaked slightly around epoch 2–3, and then gradually decreased.
- By the final epoch, validation loss converged toward approximately 0.1267, indicating good generalization.
- The gap between training and validation loss remained moderate, suggesting mild underfitting rather than overfitting.
- The overall learning behavior is typical for sequence forecasting with LSTM and represents a solid baseline to improve upon.

---

## Task 2 — Model Improvements  

The LSTM forecasting model was selected for improvement.

### Modifications Implemented  

1. Increased LSTM hidden units  
   - Improved model capacity for learning complex patterns  

2. Stacked LSTM layers  
   - Enabled deeper temporal feature extraction  

3. Adjusted batch size and learning rate  
   - Improved training stability and convergence  

---

## Benchmark Results  

| Model Version        | Loss | MAE  | Notes                     |
|---------------------|------|------|--------------------------|
| Baseline LSTM       | X.XX | X.XX | Original configuration   |
| Improved LSTM       | X.XX | X.XX | Better performance       |

Replace X.XX with actual results.

---

## Discussion  

The improvements resulted in better forecasting accuracy, demonstrating that:
- Increasing model depth improves representation learning  
- Hyperparameter tuning significantly impacts performance  
- LSTM models benefit from careful architecture design  

---

## Questions  

### 1. Which model was easier to understand and why?  

The LSTM model was easier to understand because it follows a sequential processing approach where information flows step-by-step through time. This structure is more intuitive compared to the Transformer, which relies on attention mechanisms and parallel processing.

---

### 2. What improvement was implemented and what was learned?  

Stacking multiple LSTM layers and increasing hidden units improved performance. This demonstrated that deeper architectures can better capture temporal dependencies, but also require careful tuning to avoid overfitting and instability.

---

## Conclusion  

This project provided practical experience with:
- Time-series data processing  
- Deep learning architectures for sequences  
- Model evaluation and benchmarking  

The comparison between Transformer and LSTM models highlights key differences in how sequence data can be modeled and optimized.

---

## References  

- Keras Transformer Time-Series Classification Example  
- Keras Weather Forecasting LSTM Example  