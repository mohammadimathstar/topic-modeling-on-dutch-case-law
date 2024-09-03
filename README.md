# Topic Modeling of Dutch Case Law on Eviction

This repository provides a comprehensive guide and the necessary tools for performing topic modelling on Dutch case law related to eviction. The primary goal is to identify underlying themes in the text using `Latent Dirichlet Allocation (LDA)`, a popular topic modelling technique. The analysis is presented in a step-by-step Jupyter Notebook, making it accessible even to those with limited technical background.

## Contents
__Jupyter Notebook:__

- `topic_modeling.ipynb`: This notebook walks you through the entire process of topic modelling, from data preprocessing to visualizing the results. It contains explanations and code blocks that can be run interactively.

__Python Scripts:__

- `preprocessing.py` and `utils.py`: They include several Python functions and scripts that handle various tasks, such as data preprocessing, topic modeling, and visualization.

__Data:__

- data/: This directory contains the necessary input data files. Note that the cleaned texts (after preprocessing) are saved in this directory.


__Models:__

- models/: Trained LDA models with different numbers of topics are saved here.

__Visualizations:__

- pics/: This directory contains visualizations such as top words (of each topic) and coherence plots for different topic models.


## How to Use

### 1. Set Up the Environment:

Make sure you have Python installed along with Jupyter Notebook. You will also need to install the required Python libraries by running:

`
pip install -r requirements.txt
`

### 2. Open the Jupyter Notebook:

Launch the notebook by navigating to the repository's directory and running:

`
jupyter notebook topic_modeling.ipynb
`

### 3. Follow the Steps in the Notebook:

The notebook is organized into sections that guide you through:

- Loading and preprocessing the data.
- Preparing the text for topic modelling (e.g., creating a Bag of Words model).
- Applying the LDA algorithm for a different number of topics.
- Visualizing the topics and interpreting the results.

  
### 4. Understanding the Results:

The notebook will generate visual outputs, such as coherence plots to help determine the best number of topics and top words that show the most important words in each topic. These visual aids make interpreting the topics in the case law easier.


### Key Functions
- prepare_topic_modeling_corpus: Prepares the text data by converting it into a format suitable for topic modeling.
- generate_topic_distributions: Computes the distribution of topics for each document.
- visualize_topics: Creates visual representations of the topics, including word clouds and interactive HTML visualizations.


### Target Audience

This repository is designed for legal scholars who are interested in exploring and identifying patterns in legal texts using topic modelling. No prior experience in programming or data science is required, as the notebook provides a guided and straightforward approach.

### Contact

If you have any questions or need further assistance, please feel free to contact me at `mohammadimathstar@gmail.com`.
