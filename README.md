# StockFactorExtractionUsingNeuralNetwork

## File Structure

### Code

- `dataPre.ipynb`: Download and prepare data.
- `modeling.py`: Train model.
- `feature_extraction(结果).ipynb`: Extract features, validation, and backtest.

### Model Files

- `scaler.joblib`: Model for standardizing original features for training the neural network.
- `feature_extraction.h5`: Trained feature extraction model.
- `test.pkl`, `train.pkl`, `val.pkl`: Data files generated from `dataPre.ipynb`.
- `requirements.txt`: List of libraries and dependencies required.
