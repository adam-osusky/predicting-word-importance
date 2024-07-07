# Predicting Word Importance

## About The Project

This repository contains the accompanying code for the Bachelor thesis: "Predicting Word Importance Using Pre-trained Language Models." The code is designed for training models that can predict word importance through a self-supervised learning approach.

### Project Overview

The training process involves a self-supervision technique where new words are artificially inserted into the text. The model is then trained to identify which words were inserted. The underlying idea is that the model will assign a higher likelihood of insertion to less important words, thereby learning to distinguish word importance.

### Built With

- [Hugging Face Datasets](https://github.com/huggingface/datasets): A library for accessing and sharing datasets easily.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): A library providing general-purpose architectures for NLP.
- [PyTorch](https://github.com/pytorch/pytorch): An open-source machine learning library based on the Torch library.


### Key Features

- **Text Dataset Creation**: The repository includes job for creating a text dataset with inserted words. There are two insertion methods:
  - **List Inserting Method (LIM)**: Randomly selects words from the original dataset for insertion.
  - **BERT Inserting Method (BIM)**: Uses BERT's masked language modeling capabilities to determine which words to insert.
  - For both methods, insertion positions are randomly chosen within the text.
- **Model Training**: Comprehensive job for training models on the created dataset. The training framework leverages the transformers library and allows for extensive customization of training parameters, such as learning rate, batch size, and number of epochs. 
- **Evaluation**: Evaluation of trained models using the metrics proposed in the thesis. These metrics focus on ranking objects, specifically an incomplete ranking for the top 10% most important words. The similarity of these rankings can be measured using Pearson correlation. Additional metrics include k-inter, which measures the number of rankings with intersections greater than k, and [average overlap](https://dl.acm.org/doi/abs/10.1145/1852102.1852106), a metric for computing the similarity of two ranking lists.


## Getting Started

### Installation
Follow these steps to set up the project on your local machine:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/adam-osusky/predicting-word-importance.git
   ```
2. **Initialize the virtual environment:**
   ```sh
   cd predicting-word-importance/
   python -m venv env
   . env/bin/activate
   ```
3. **Install the package**
   ```
   pip install .
   ```
By following these steps, you'll set up a virtual environment and install all necessary dependencies for the project.


## Usage
This repository provides various jobs for creating datasets by inserting new words into WikiText, training models, and evaluating the trained models. These jobs can be easily parameterized using JSON configuration files.

### Dataset Creation
To create a dataset with inserted words, follow these steps:

1. Create a json file `ds.json`:
    ```json
    {
        "job": "WikiTextDsJob",
        "insert_rate": 0.5,
        "insert_model": "google-bert/bert-base-uncased"
    }
    ```
   - `job`: Specifies the type of job to run.
   - `insert_model`: Specifies the model for word insertion. You can use any HuggingFace model suitable for masked language modeling, or "random" to select words randomly from the base text corpus.
   - for now the base text corpus is fixed and it is a [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) 
   - `insert_rate`: Specifies the rate at which new words are inserted. For example, 0.5 means approximately 50% of the words in each sample will be new insertions.

1. Run the dataset creation job:
    ```sh
    python -m predwordimp.util.run ds.json 
    ```
    - The dataset will be saved to `./data/wikitext/<job-name>` directory, where the job name is created from a timestamp and a randomly chosen adjective and noun. In this diroctory will be the dataset saved in HF format and also the configuration json will be saved here too. This property has every job and it is usefull for experiment tracking.

### Training Job
To train a model on the created dataset, follow these steps:
1. Create a JSON file `train.json` with the following content:
    ```json
    {
        "job": "TrainJob",
        "model": "google-bert/bert-base-uncased",
        "dataset_dir": "./data/wikitext/20240310130913_red_medved",
        "batch_size": 32,
        "num_proc": 10,
        "hf_access_token": null,
        "epochs": 5,
        "warmup_steps": 300,
        "logging_steps": 50,
        "eval_steps": 300,
        "save_strategy": "epoch",
        "save_total_limit": 5,
        "gradient_accumulation_steps": 8
    }
    ```
    - `model`: Specifies the pre-trained model to fine-tune.
    - `dataset_dir`: Path to the dataset created in the previous step.
    - Other parameters are standard options for the transformers' Trainer object, such as batch_size, epochs, and save_strategy etc.
2. Run the training job:
    ```sh
    python -m predwordimp.util.run train.json 
    ```
    - The configuration and model checkpoints will be saved to `./data/train/<job-name>`, where the job name is created similarly as in the previous job.

### Evaluation of Trained Models
To evaluate a trained model, follow these steps:
1. Create a JSON file `eval.json` with the following content:
    ```json
    {
        "job": "EvalWordImp",
        "hf_model": "adasgaleus/LIM-0.25"
    }
    ```
    - `hf_model`: Path to the model created in the previous training job.
2. Run the evaluation job:
    ```sh
    python -m predwordimp.util.run eval.json 
    ```
    - The configuration used and the results will be saved to `./data/eval/<job-name>`
    - The results will be in JSON format and include metrics such as Pearson correlation, k-inter, and overlap.

### Reproduce main results
To reproduce the main evaluation results from the thesis, execute the following command:
```sh
python -m predwordimp.eval.reproduce.evaluate
```
This command will download six BERT base models, which are the results of the thesis, and evaluate them on the [Word Importance Dataset](https://huggingface.co/datasets/adasgaleus/word-importance). Below are the results from the thesis:

| Model      | Pearson | 1-inter | 2-inter | 3-inter | 4-inter | 5-inter | Overlap |
|------------|---------|---------|---------|---------|---------|---------|---------|
| Random     | 0.256   | 0.54    | 0.13    | 0.01    | 0.00    | 0.00    | 0.061   |
| PI         | 0.321   | 0.78    | 0.40    | 0.08    | **0.04**| 0.00    | 0.114   |
| TF-IDF     | 0.309   | 0.66    | 0.20    | 0.04    | 0.00    | 0.00    | 0.121   |
| BIM-0.75   | 0.335   | 0.82    | 0.32    | 0.12    | 0.02    | 0.00    | 0.125   |
| BIM-0.25   | 0.341   | 0.76    | 0.40    | 0.14    | 0.02    | **0.02**| 0.131   |
| LIM-0.5    | 0.328   | 0.72    | 0.40    | 0.12    | **0.04**| 0.00    | 0.137   |
| LIM-0.75   | 0.352   | 0.80    | 0.48    | 0.18    | **0.04**| 0.00    | 0.142   |
| BIM-0.5    | 0.344   | 0.70    | 0.42    | 0.14    | 0.02    | **0.02**| 0.143   |
| NLI        | 0.374   | **0.90**| **0.56**| **0.22**| **0.04**| **0.02**| 0.150   |
| LIM-0.25   | **0.376**| 0.82   | 0.52    | 0.14    | 0.02    | **0.02**| **0.178**|

Where PI and NLI are models from [Assessing Word Importance Using Models Trained for Semantic Tasks](https://arxiv.org/abs/2305.19689)


## Project Structure

- `src/predwordimp/util`
  - Contains utility scripts. Notably, `job.py` includes the main base class for all other jobs, and `run.py` is used for executing these jobs.

- `src/predwordimp/data/dataset_job.py`
  - Contains the job for creating a training dataset. Currently, it is hardcoded to use WikiText as the base text for word insertion.

- `src/predwordimp/training/train.py`
  - Contains the job for training the models. Prepares the data with tokenization and aligning of labels. the training loop is done with HF Trainer.

- `src/predwordimp/eval`
  - Module for evaluation.
  - `metrics.py` implements the metrics described in the thesis.
  - `wi_eval.py` includes the evaluation job for models trained by self-supervision, as well as for TF-IDF.
  - `kpe.py` is a WIP evaluation of the models on key-phrase extraction.
  - `annotation.py` provides helper code for analyzing collected annotations.
  - `reproduce` module is used for reproducing the main evaluations from the thesis.