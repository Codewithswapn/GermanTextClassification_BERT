# GermanTextClassification_UsingBERT

This project implements a text classification model to classify German news articles into 9 different categories using the GNAD10 dataset. The model is based on the BERT architecture and was developed using Huggingface's Transformers library and PyTorch.

## Project Overview

The goal of this project is to build and evaluate a model capable of accurately classifying news articles written in German. This project is structured into several sections to guide you through the data loading, preprocessing, model training, evaluation, and potential future work.

## Dataset

The dataset used is the **GNAD10** dataset, which consists of 10,000 German news articles categorized into 9 distinct classes. The dataset is available on Huggingface Datasets and is divided into training and test sets.

- **Dataset Source**: [GNAD10 on Huggingface](https://huggingface.co/datasets/gnad10)

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Huggingface)
- Datasets (Huggingface)
- Evaluate (Huggingface)
- NLTK
- Matplotlib
- Pandas
- WordCloud

You can install the required libraries using pip:

```bash
pip install torch transformers datasets evaluate nltk matplotlib pandas wordcloud
```

## Project Structure

- **Data Loading and Preprocessing**:
  - The dataset is loaded using the Huggingface Datasets library.
  - Preprocessing includes lowercasing, removing punctuation, tokenization, and stopwords removal using NLTK.
  
- **Model Training**:
  - A BERT-based model (`bert-base-german-cased`) is fine-tuned on the dataset.
  - The training loop includes tracking the loss, accuracy, and F1 score.

- **Evaluation**:
  - The model's performance is evaluated using overall accuracy and F1 score.
  - Additionally, per-class accuracy and F1 score are analyzed to identify class-specific performance.

- **Visualization**:
  - Word clouds are generated for each class to visualize the most common words in the dataset.

- **Future Work**:
  - Exploring long-range BERT models like Longformer to handle texts longer than 512 tokens.
  - Hyperparameter tuning using tools like Optuna or Raytune to optimize model performance.

## Results

- **Accuracy**: The model achieved an overall accuracy of 90.85%.
- **F1 Score**: Detailed per-class F1 scores are provided in the notebook.

The model shows good performance but could be further improved by fine-tuning hyperparameters or exploring different architectures.

## Future Improvements

1. **Hyperparameter Tuning**: Implementing hyperparameter optimization to identify the best configuration for the model.
2. **Long-Range Models**: Testing models that can handle longer text inputs, like Longformer.
3. **Increased Validation Size**: Experimenting with different validation set sizes to improve model generalization.

