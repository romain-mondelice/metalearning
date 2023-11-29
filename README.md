# Meta-Learning for Time Series Forecasting

## Introduction
This project explores the application of meta-learning to time series forecasting, with a focus on predicting Bitcoin prices. The main goal is to leverage a meta-learning approach to improve the training process and forecasting accuracy of a neural network-based model.

## Methodology

### Data Preprocessing
- **Dataset**: Historical Bitcoin price data.
- **Features**: Various indicators such as "Number of Transactions", "Difficulty", "Total Supply", etc.
- **Target**: Predicting the next 12 closing prices.

### Base Model
- **Architecture**: Neural network tailored for time series forecasting (LSTM).
- **Training**: Conducted using the Adam optimizer and Mean Squared Error (MSE) loss.

### Meta Model (Learner Network)
- **Purpose**: To learn optimal parameter updates for the base model.
- **Input**: Gradients from the base model.
- **Output**: Predicted parameter updates.
- **Training Methodology**: Trained on the difference between actual and predicted parameter updates of the base model. We take the updated parameters of the base model (after the .step()) and set it as a target for the meta-learner.
The objective is to learn of the base model learn.

### Training Process
- **Dual Training**: Simultaneous training of both the base and meta models.
- **Loss Function**: Mean Squared Error (MSE) for both models.

## Results

### Basic Training (Training normaly with a classic training loop and loss function)
- **Same Objective that during the meta learner training process (predict the 12 next close prices)**:
  - Training MSE Loss: 0.9518
  - Test MSE Loss: 1.0210
  - Training Time: 108.13 seconds
- **Mid Price Objective (predict the 12 next mid price)**:
  - Training MSE Loss: 0.9479
  - Test MSE Loss: 1.0280
  - Training Time: 90.68 seconds

### Meta Training
- **Same Objective that during the meta learner training process (predict the 12 next close prices)**:
  - Training MSE Loss: 0.9527
  - Test MSE Loss: 1.0254
  - Training Time: 4.51 seconds
- **Mid Price Objective (predict the 12 next mid price)**:
  - Training MSE Loss: 0.9527
  - Test MSE Loss: 1.0258 (Better loss, good sign of the capability to reduce overfitting)
  - Training Time: 5.14 seconds

## Discussion
- The experiment demonstrates the feasibility of applying meta-learning in time series forecasting.
- The meta model shows a comparable performance in predicting parameter updates, with significantly reduced training times compared to basic training.
- Challenges in the project included managing the complexity of dual-model training and ensuring the correct application of predicted gradients.

## Future Work
- Feature engineering
- Further refinement of the base model's architecture (try more complexe archi / more simple).
- Further refinement of the meta model's architecture (try more complexe archi / more simple).
- Finetuning of hyperparameters (learning rates & num_epoch in the first place).

- **Train the meta learner on several objective in the first place (the objective is here to be able to learn how to learn, so more objective seen, the better in my opinion, also good for generalization and reduce overfitting).**

- Extending the approach to other time series datasets.

## Conclusion
This project represents an innovative approach to time series forecasting, where meta-learning is employed to enhance the training efficiency and potentially the accuracy of a forecasting model. The results indicate promising directions for future research in this area.
