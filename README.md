# Zero-Shot Text Classification with Newsgroups Dataset

This notebook demonstrates an approach to zero-shot text classification using a subset of the 20 Newsgroups dataset. It leverages a pre-trained multilingual encoder along with a trainable projection layer to map input texts to a candidate label space. The model dynamically computes candidate label embeddings during each forward pass, allowing the projection layer to update effectively through training.

## Features

- **Pre-trained Multilingual Encoder:** Uses DistilBERT to extract deep contextual features from text.
- **Dynamic Candidate Label Embedding:** Computes embeddings for candidate labels during each forward pass for better integration with the model.
- **Learning Rate Scheduler:** Implements a scheduler to adjust the learning rate when progress slows down.
- **Custom Data Pipeline:** Utilizes scikit-learn's 20 Newsgroups dataset and processes the text data for training.
- **Comprehensive Visualizations:** Includes training loss curves, confusion matrix heatmaps, and t‑SNE plots to visualize the model’s behavior.

## Requirements

- Python 3.6+
- PyTorch
- Transformers (by Hugging Face)
- scikit-learn
- Matplotlib
- Seaborn
- numpy

## Installation

1. **Clone the repository:**

   ```bash
    git clone https://github.com/neuralsorcerer/zsl.git
    cd zsl
    ```
2. **Install the required packages:**

   ```bash
   pip install torch transformers scikit-learn matplotlib seaborn numpy
    ```
    If you encounter dependency conflicts, consider using a virtual environment or conda environment.
    
## Usage

1. **Open the Notebook:**

    Launch Jupyter Notebook or JupyterLab and open the provided notebook file.

2. **Run the Notebook Cells:**

    Execute the cells in order to load the dataset, train the model, and generate visualizations. The notebook is divided into clear sections for data loading, model building, training, evaluation, and visualization.

3. **Experiment and Modify:**

    - Adjust the number of training epochs.

    - Modify batch sizes or learning rates.

    - Explore alternative candidate labels or additional preprocessing steps.

    - Review the visualizations to gain insights into the model’s performance.
    
## Dataset

This notebook uses a subset of the 20 Newsgroups dataset, focusing on four categories:

- `rec.sport.baseball` (mapped to Sports)

- `talk.politics.mideast` (mapped to Politics)

- `sci.space` (mapped to Sci/Tech)

- `comp.sys.mac.hardware` (mapped to Technology)

The data is loaded using scikit-learn’s built-in functions and is split into training and testing sets.

## Model Architecture

The model consists of:

- **Encoder:** A pre-trained DistilBERT model that extracts contextual embeddings from the input text.

- **Projection Layer:** A trainable linear layer that maps the encoder’s output to a space compatible with candidate label representations.

- **Dynamic Candidate Label Embedding:** Candidate label embeddings are computed on-the-fly during the forward pass, ensuring that the projection layer is updated effectively.

## Training and Evaluation

The training loop includes:

- Mini-batch training using PyTorch's DataLoader.

- Use of the Adam optimizer.

- A ReduceLROnPlateau learning rate scheduler to adaptively reduce the learning rate if the loss stops improving.

- Calculation of evaluation metrics such as accuracy and weighted F1 score.

Evaluation is performed on a held-out test set, and the performance is reported through a classification report and confusion matrix.

## Visualizations

The notebook generates several key visualizations to help understand the model's performance:

- **Training Loss Curve:** Shows how the loss decreases (or plateaus) over epochs.

- **Confusion Matrix:** A heatmap that displays the distribution of true vs. predicted labels.

- **t‑SNE Plot:** Visualizes high-dimensional embeddings in two dimensions, highlighting the relationship between test text embeddings and candidate label embeddings.

## License
This project is licensed under the [MIT License](LICENSE).
