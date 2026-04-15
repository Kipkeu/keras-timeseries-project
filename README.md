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


## Project Structure

```
project/
│
├── README.md
│
└── results/
    │
    ├── Transformer_baseline/
    │       test_results.txt
    │       transformer_accuracy.png
    │       transformer_loss.png
    │
    ├── LSTM_baseline/
    │       lstm_loss.png
    │       prediction_0.png
    │       prediction_1.png
    │       prediction_2.png
    │       prediction_3.png
    │       prediction_4.png
    │       training_history.txt
    │
    └── LSTM_Improvements/
        │
        ├── Improvement_1_LR_Scheduler/
        │       lstm_loss.png
        │       prediction_0.png
        │       prediction_1.png
        │       prediction_2.png
        │       prediction_3.png
        │       prediction_4.png
        │       training_history.txt
        │
        ├── Improvement_2_Dropout/
        │       lstm_loss.png
        │       training_history.txt
        │
        └── Improvement_3_Stacked_LSTM/
                lstm_loss.png
                prediction_0.png
                prediction_1.png
                prediction_2.png
                prediction_3.png
                prediction_4.png
                training_history.txt
```

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
**Training Accuracy:** 51.7%  
**Validation Accuracy:** 49.2%  
**Test Accuracy:** 51.6

**Notes / Observations:**

- The model did not learn meaningful patterns from the FordA dataset.
- Training and validation accuracy remained near chance level (~50%) across all epochs.
- Loss stayed approximately constant around 0.693, indicating random guessing behaviour.
- Validation accuracy plateaued early with minor fluctuations but no sustained improvement.
- Overall performance indicates clear under-fitting.


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

### Overall: Which model was easier to understand and why?  
The LSTM model was easier to understand because it follows a sequential processing approach where information flows step-by-step through time. This structure is more intuitive compared to the Transformer, which relies on attention mechanisms and parallel processing. In addition the Transformer model on the FordA dataset showed near-random performance, with accuracy remaining close to 50%. This indicates that the model failed to learn meaningful patterns under the default configuration. This behavior is often observed when Transformers are not sufficiently tuned or when the dataset is not large or complex enough to benefit from attention mechanisms. In contrast, the LSTM model demonstrated stable and consistent learning on the forecasting task. Training and validation loss decreased smoothly, indicating that the model successfully captured temporal dependencies in the data. This made the LSTM a more suitable candidate for controlled experimentation and performance benchmarking.

The LSTM model was thus selected for further improvements due to:

- Reliable convergence behaviour  
- Clear learning trends in loss curves  
- Strong baseline performance  
- Greater interpretability of architectural changes  

For the LSTM forecasting task, loss refers to Mean Squared Error (MSE), and Mean Absolute Error (MAE) is reported where relevant to assess forecasting accuracy.

---

## Task 2 — LSTM Improvements: What improvement was implemented and what was learned?   

#### Experiment 1: Incorporating Learning Rate Scheduler

Objective: Improve convergence stability during training.

Added ReduceLROnPlateau callback to dynamically reduce learning rate when validation loss plateaus.

### Results

| Metric                     | Baseline LSTM        | Experiment 1 (LR Scheduler) |
|--------------------------|---------------------|-----------------------------|
| Initial Learning Rate     | 0.001               | 0.001 → 0.0005 (adaptive)   |
| Training Loss (final)     | ~0.10               | ~0.0976                     |
| Validation Loss (best)    | ~0.1267             | **~0.1114**                 |
| Convergence Behavior      | Smooth but plateau  | Plateau → improved after LR drop |
| Overfitting               | Mild                | Reduced                     |
| Stability (val_loss)      | Moderate fluctuation| More stable after LR reduction |
| Training Efficiency       | Fixed rate          | Adaptive optimization       |

- Initial training showed rapid improvement in validation loss up to epoch 1.
- After plateauing, validation loss temporarily stagnated and increased slightly.
- Once the learning rate was reduced, performance improved significantly.
- Best validation loss improved from approximately 0.1267 (baseline) to 0.1114.

The adaptive learning rate allowed the optimizer to escape a plateau and refine convergence in later epochs. This resulted in both improved stability and better generalization performance.


#### Experiment 2: Dropout Regularization  

Objective: Reduce overfitting and improve generalization.

Increased dropout rate in LSTM layers and introduced recurrent dropout to regularise training.

### Results

| Model / Experiment        | Best Val Loss | Performance Change | Interpretation |
|--------------------------|--------------|-------------------|----------------|
| Baseline LSTM            | 0.1267       | —                 | Stable convergence with mild underfitting |
| Experiment 1 (LR Scheduler) | **0.1114** | Improved          | Best optimisation and convergence |
| Experiment 2 (Dropout)   | 0.1366       | Trade-off         | Higher loss but lowest MAE (more robust predictions) |
| Experiment 3 (Stacked LSTM) | TBD        | TBD               | Expected to improve temporal representation |

- Best Validation Loss: 0.1366  
- Final Validation MAE: **0.2923**  

### Observations

- Training loss started higher than the baseline due to the regularisation effect of dropout.
- Loss decreased more gradually, indicating slower but more controlled learning.
- Validation loss improved steadily across epochs, with fewer sharp oscillations.
- A learning rate reduction at later epochs contributed to additional improvements.

### Interpretation

The addition of dropout reduced overfitting by preventing the model from relying too heavily on specific neurons. While the validation loss increased compared to the baseline and Experiment 1, the model achieved the lowest MAE across all experiments.

This indicates that:
- Dropout improves robustness to noise and outliers  
- It introduces a bias-variance trade-off  
- Lower MAE suggests better average prediction accuracy, even if squared error increases  

Overall, dropout produces more stable and reliable predictions, but does not yield the best performance in terms of validation loss.

#### Experiment 3: Model Capacity Adjustment — Stacked LSTM

Objective: Evaluate the impact of increased model capacity on temporal feature learning.

Instead of simply increasing the number of hidden units, the architecture was modified by stacking two LSTM layers (64 → 32 units). This allows the model to learn hierarchical temporal representations:
- The first layer captures low-level temporal patterns
- The second layer refines higher-level dependencies

### Results

| Metric | Value |
|--------|-------|
| Model Type | Stacked LSTM (64 → 32 units) |
| Best Epoch | 9 |
| Final Training Loss | ~0.1014 |
| Best Validation Loss | 0.1534 |
| Final Validation MAE | ~0.3087 |
| Total Parameters | 31,393 |

### Observations

- Training loss improved compared to the baseline, indicating increased learning capacity.
- However, validation loss worsened significantly, suggesting overfitting.
- The model learned more complex patterns but failed to generalise well to unseen data.
- Increased model depth introduced additional parameters, making optimisation more difficult without stronger regularisation.

This demonstrates that increasing model complexity does not guarantee better performance, especially for moderately sized datasets.

## Benchmark Results    

| Experiment | Modification | Best Val Loss | Validation MAE | Key Insight |
|------------|-------------|---------------|----------------|-------------|
| Baseline | Single LSTM (32 units) | 0.1267 | ~0.32 | Stable learning, mild underfitting |
| Exp 1 | LR Scheduler | **0.1114** | ~0.31 | Best convergence and optimisation |
| Exp 2 | Dropout | 0.1366 | **0.2923** | Best MAE, improved robustness |
| Exp 3 | Stacked LSTM | 0.1534 | 0.3081 | Overfitting due to increased complexity |
---

## Key Insights

- Adaptive learning rates (Experiment 1) provided the most significant improvement in validation loss, indicating better optimisation and convergence.
- Dropout (Experiment 2) improved prediction robustness, achieving the lowest MAE despite a slightly higher validation loss.
- Increasing model complexity (Experiment 3) led to worse performance, suggesting overfitting and reduced generalisation.
- Simpler models with appropriate regularisation outperform more complex architectures on this dataset.

---
  
## Discussion  

The experiments highlight the importance of controlled model tuning in time-series forecasting. While increasing model capacity improved training performance, it did not translate to better validation results. This suggests that the dataset does not necessarily require a highly complex model and that additional parameters lead to overfitting rather than improved feature extraction. In contrast, optimisation strategies such as learning rate scheduling had a significant impact on performance. By allowing the model to adjust its learning dynamics during training, the optimiser was able to escape local minima and achieve better convergence. Regularisation through dropout further stabilised training, reducing variance and improving generalisation, although it did not outperform the learning rate scheduler in terms of loss.

--- 

## Conclusion  

This project successfully reproduced and evaluated two deep learning approaches for time-series tasks: Transformer-based classification and LSTM-based forecasting. 

The Transformer model failed to learn effectively under default settings, highlighting the sensitivity of attention-based models to hyperparameter tuning and dataset characteristics. The LSTM model demonstrated strong baseline performance and was selected for further improvement. Through systematic experimentation, it was found that the best-performing model was the LSTM with adaptive learning rate (Experiment 1), achieving the lowest validation loss. Across all experiments, performance improvements were driven more by optimisation strategies (learning rate scheduling) than architectural complexity increases, highlighting the importance of training dynamics over model size for this dataset.

This study emphasises the importance of optimisation and regularisation over blindly increasing model complexity when working with time-series data.