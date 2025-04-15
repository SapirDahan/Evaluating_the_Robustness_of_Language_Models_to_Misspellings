# Robustness Analysis of Language Models on Noisy Questions

## Project Overview
This research project investigates how a diverse set of over 20 language models from HuggingFace respond to input questions that have been intentionally perturbed by misspellings. The primary goal is to assess the robustness of these models when faced with noisy, real-world data.

## Project Purpose
Language models are widely used for various natural language processing tasks. However, in real-world settings, users often submit inputs that contain typographical errors or misspellings. This project aims to:
- Evaluate the impact of misspellings on the quality of model-generated outputs.
- Compare and contrast the performance of different model architectures (e.g., small vs. large models, standard vs. specialized models).
- Provide detailed empirical insights using multiple similarity metrics and advanced statistical analyses.
- Generate a wide range of visualizations and evaluation tables to help understand how each model's output degrades (or remains robust) as the level of noise increases.

## How the Project Works
The project workflow is divided into several steps:

1. **Data Preparation**
   - **Input Datasets:**  
     - A questions dataset (a CSV file with a single column named `question`) containing a large number of questions.
     - A misspelling dataset (sourced from Kaggle) that lists common correct words along with their typical misspellings.
   - **Augmentation:**  
     The project automatically generates numerous variants of each question by introducing 0, 1, …, up to N misspellings. The augmentation function dynamically limits the number of errors according to the number of words in each question that are eligible for replacement.
  
2. **Model Inference**
   - Over 20 pre-trained language models are loaded via HuggingFace’s Transformers library.
   - Each model processes every augmented question variant, and their respective outputs are stored along with metadata such as the original question, the noise level (error count), and the specific model that produced the output.

3. **Results Analysis**
   - **Similarity Metrics:**  
     The project computes multiple metrics to compare the model outputs on clean (error-free) questions to those on noisy (misspelled) questions. These include:
     - Difflib similarity (0–100 scale)
     - Cosine similarity of sentence embeddings (using Sentence-Transformers)
     - BLEU score (using NLTK)
     - Jaccard similarity (based on token overlap)
   - **Visualizations:**  
     An extensive set of graphs is created, including boxplots, histograms, density plots, scatter plots (grouped by sentence length), global line charts, correlation heatmaps, and pair plots. This visualization suite is intended to comprehensively display the impact of increasing noise on the model outputs.

4. **Deep Evaluation**
   - In addition to visualizations, detailed evaluation tables are generated. These tables report statistics such as the mean, standard deviation, median, and quartiles of the similarity scores for each model.
   - Models are also grouped and compared by class (e.g., Small, Medium, Large, Standard) based on their architecture or naming.
   - Advanced statistical analyses (e.g., regression analysis, t-tests, correlation analyses) are performed to deepen the research insights.

## How to Set Up and Run the Project
1. **Installation:**
   - Ensure you have Python 3.7 or later and a strong GPU available.
   - Install the required dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - Download the necessary NLTK data by running:
     ```python
     import nltk
     nltk.download('punkt')
     ```

2. **Data Preparation:**
   - Place your questions dataset in a file named `data/questions.csv`. This CSV should have a header with a column called `question`.
   - Download the Kaggle Common English Misspellings dataset and save it as `data/misspellings.csv` (with columns `correct_word` and `misspellings`).
   - Run the Data Preparation notebook (e.g., **Data_Preparation.ipynb**) to generate the augmented questions file (`data/augmented_questions.csv`).

3. **Model Inference:**
   - Run the Model Inference notebook (e.g., **Model_Inference.ipynb**) to process the augmented questions with over 20 models. The outputs will be stored in `data/model_outputs.csv`.

4. **Results Analysis and Deep Evaluation:**
   - Run the Results Analysis notebook (e.g., **Results_Analysis.ipynb**) to compute similarity metrics and generate a wide variety of visualizations.
   - Finally, run the Deep Evaluation notebook (e.g., **Deep_Evaluation.ipynb**) to produce detailed evaluation tables and perform additional statistical analyses.

## Extensibility
- **Additional Noise Types:** The modular design makes it easy to add other types of noise (e.g., grammatical errors, punctuation errors).
- **New Evaluation Metrics:** More metrics can be easily integrated into the evaluation pipeline.
- **Interactive Dashboards:** Future work might include interactive dashboards for real-time analysis of model performance.
  
## Acknowledgements
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- The creators of the Common English Misspellings dataset on Kaggle.
- All research papers and resources that inspired this project.

Happy researching!"# Evaluating_the_-Robustness_of_Language_Models_to_Misspellings" 
