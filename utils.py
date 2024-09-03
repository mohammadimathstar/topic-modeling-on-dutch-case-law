import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from preprocessing import STOP_WORDS, nlp

from gensim import corpora
from gensim.test.utils import datapath
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis 

def load_eviction_case_identifiers(file_path="./data/20240418 ALL 5047 eviction cases.csv"):
    """
    Load the ECLI (European Case Law Identifier) codes from a CSV file containing eviction cases.
    
    Parameters:
        file_path (str): The file path to the CSV containing eviction case data. 
                         The default is './data/eviction.csv'.
    
    Returns:
        list: A list of ECLI codes for the eviction cases.
    """
    # Load the eviction cases from the CSV file, specifically the 'ECLI' column
    eviction_df = pd.read_csv(file_path, delimiter=";", usecols=["ECLI"])
    
    # Convert the 'ECLI' column to a list
    eviction_ecli_list = eviction_df['ECLI'].tolist()
    
    # Print the total number of eviction cases found
    print(f"There are {len(eviction_ecli_list)} ecli numbers of eviction cases.")
    
    # Return the list of ECLI codes
    return eviction_ecli_list

def extract_eviction_case_text(eviction_ecli_list, file_path="./data/2000-2023.pickle"):
    """
    Extract the judgment texts of eviction-related cases based on a list of ECLI codes.
    
    Parameters:
        eviction_ecli_list (list): A list of ECLI codes related to eviction cases.
        file_path (str): The file path to the pickle file containing the full case data.
                         The default is './data/2000-2023.pickle'.
    
    Returns:
        tuple: A tuple containing:
            - ecli_found (list): A list of ECLI codes found in the dataset.
            - eviction_corpus (list): A list of judgment texts corresponding to the found ECLI codes.
    """
    # Load the dataset from the pickle file
    case_df = pd.read_pickle(file_path)
    
    # Filter the DataFrame to include only rows with ECLI codes in the provided list
    ecli_found = case_df.index[case_df.index.isin(eviction_ecli_list)].tolist()
    
    # Extract the judgment texts for the selected ECLI codes
    eviction_corpus = case_df.loc[ecli_found, 'judgment_text'].tolist()
    
    # Print the number of eviction-related cases found with texts
    print(f"There are {len(ecli_found)} eviction-related cases with texts ({len(eviction_ecli_list)-len(ecli_found)} cases with no text).")
    
    # Return the found ECLI codes and their corresponding judgment texts
    return ecli_found, eviction_corpus


def clean_texts(ecli_nos, texts, file_path=None):
    """
    Clean and preprocess a list of legal texts by lemmatizing, removing stop words, 
    and filtering out unwanted tokens (e.g., short words, non-alphabetic characters, named entities).
    
    Parameters:
        ecli_nos (list): A list of ECLI codes corresponding to the texts.
        texts (list): A list of legal texts to be cleaned.
        file_path (str, optional): File path to save the cleaned texts and ECLI codes.
                                   If None, the cleaned texts are not saved.
    
    Returns:
        list: A list of ECLI numbers.
        list: A list of cleaned and preprocessed texts.
    """
    cleaned_texts = []
    
    # Use tqdm to create a progress bar for the cleaning process
    for txt in tqdm(texts, desc="Cleaning texts"):
        cleaned_texts.append(
            " ".join([
                token.lemma_.lower() for token in nlp(txt) if (
                    token.lower_ not in STOP_WORDS
                ) and (
                    token.lemma_.lower() not in STOP_WORDS
                ) and (
                    token.is_alpha
                ) and (
                    len(token) > 3
                ) and (
                    len(token.ent_type_) == 0
                )
            ])
        )
    
    # Save cleaned texts and ECLI codes if file path is provided
    if file_path is not None:
    	df = pd.DataFrame({"ecli": ecli_nos, "clean_text": cleaned_texts})
    	df.to_csv(file_path, index=False) 
    	print(f"Cleaned texts and ECLI codes saved to {file_path}.")
    
    return ecli_nos, cleaned_texts
    
    
def prepare_topic_modeling_corpus(docs, min_doc_count=5, max_doc_proportion=0.4):
    """
    Create a dictionary and a corpus for topic modeling based on the provided preprocessed texts.
    
    Parameters:
        docs (list of list of str): A list of documents where each document is represented as a list of tokens.
        min_doc_count (int): Minimum number of documents a token must appear in to be included in the dictionary.
        max_doc_proportion (float): Maximum proportion of documents a token can appear in to be included in the dictionary.
    
    Returns:
        tuple: A tuple containing:
            - id2word (corpora.Dictionary): A Gensim Dictionary mapping of token IDs to token words.
            - corpus (list of list of tuples): A list of bag-of-words representations for each document.
    """
    # Create a dictionary from the documents
    id2word = corpora.Dictionary(docs)
    
    # Filter out tokens that occur in fewer than min_doc_count documents or in more than max_doc_proportion of the documents
    id2word.filter_extremes(
        no_below=min_doc_count,
        no_above=max_doc_proportion,
        keep_n=None  # Keep all tokens that meet the criteria
    )
    
    # Convert the list of documents into a bag-of-words format using the dictionary
    corpus = [id2word.doc2bow(doc) for doc in docs]
    
    # Print out information about the resulting dictionary and corpus
    print(f"Tokens considered: Appear in at least {min_doc_count} documents and at most {max_doc_proportion * 100}% of documents.")
    print(f"Number of unique tokens: {len(id2word)}")
    print(f"Number of documents: {len(corpus)}")
    
    return id2word, corpus
    

def create_lda_model(corpus, id2word, num_topics):
    """
    Create an LDA (Latent Dirichlet Allocation) model using Gensim.
    
    Parameters:
        corpus (list of list of tuples): A list of documents represented as bag-of-words (BoW) format,
                                         where each document is a list of tuples (word_id, frequency).
        id2word (corpora.Dictionary): A Gensim Dictionary mapping word IDs to words.
        num_topics (int): The number of topics to extract from the corpus.
    
    Returns:
        LdaModel: A trained LDA model.
    """
    
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=100,  # Seed for random number generator for reproducibility
        update_every=1,   # Update the model every iteration
        chunksize=100,    # Number of documents to be used in each training chunk
        passes=10,        # Number of passes over the entire corpus
        alpha='auto',    # Alpha parameter for the topic distribution
        per_word_topics=True  # Whether to store per-word topic probabilities
    )
    
    return model
    
    
def selection_of_number_of_topics(idx2word, doc_term_matrix, clean_text, start=5, stop=7, step=1, coherence_type='c_v'):
    """
    Evaluate the coherence score of LDA models with varying numbers of topics.
    
    Parameters:
        idx2word (corpora.Dictionary): Gensim dictionary mapping word IDs to words.
        doc_term_matrix (list of list of tuples): The corpus in bag-of-words format.
        clean_text (list of list of str): The preprocessed texts corresponding to the corpus.
        start (int): Starting number of topics for evaluation.
        stop (int): Ending number of topics (exclusive) for evaluation.
        step (int): Step size for the number of topics.
        coherence_type (str): Type of coherence measure to use. Options are 'c_v', 'c_uci', 'u_mass'.
    
    Returns:
        tuple: A tuple containing:
            - ntopics_range (list of int): The range of topic numbers evaluated.
            - coherence_values (list of float): The coherence values corresponding to each number of topics.
    """
    coherence_values = []
    ntopics_range = list(range(start, stop, step))
    
    # Define directory paths
    cwd = os.getcwd()
    models_dir = os.path.join(cwd, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f'Computing coherence scores using "{coherence_type}" coherence measure...')
    
    for num_topics in tqdm(ntopics_range):
        # Create directories and files
        num_topics_dir = os.path.join(models_dir, str(num_topics))
        os.makedirs(num_topics_dir, exist_ok=True)
        
        temp_file = os.path.join(num_topics_dir, "lda_model")
        
        # Create and save the LDA model
        model = create_lda_model(corpus=doc_term_matrix, id2word=idx2word, num_topics=num_topics)
        
        
        # Compute coherence score
        coherence_model = CoherenceModel(
            model=model,
            texts=clean_text,
            dictionary=idx2word,
            coherence=coherence_type  # Options: 'c_v', 'c_uci', 'u_mass'
        )
        
        model.save(temp_file)
        coherence_values.append(coherence_model.get_coherence())
    
    # Save coherence values to a CSV file
    coherence_df = pd.DataFrame({"num_of_topics": ntopics_range, "coherence_values": coherence_values})
    csv_file = os.path.join(models_dir, f'coherence_values_{coherence_type}.csv')
    coherence_df.to_csv(csv_file, index=False)
    
    # Plot coherence values
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=ntopics_range, 
        y=coherence_values,
        color='royalblue',
        marker='^',
        markersize=12,
        dashes=True,
        lw=2,
        label=coherence_type
    )
    
    plt.legend(loc='best', fontsize=14)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Value')
    plt.title('Coherence Values for Various Number of Topics')
    plt.tight_layout()
    
    # Save the plot as an image
    plot_file = os.path.join(cwd, f"pics/coherence_values_{coherence_type}.jpg")
    plt.savefig(plot_file, dpi=150)
    
    return ntopics_range, coherence_values
    
    
def get_word_cloud_text(weights, id2token):
    """
    Generate a dictionary of words and their corresponding frequencies for word cloud generation.
    
    Parameters:
        weights (list of float): Weights or probabilities associated with each word.
        id2token (dict): Mapping from word IDs to the actual words.
        
    Returns:
        dict: A dictionary where keys are words and values are their scaled frequencies.
    """
    return {id2token[i]: int(w * 100.0) for i, w in enumerate(weights)}


def create_word_cloud(frequencies, background_color='white'):
    """
    Create a word cloud object from word frequencies.
    
    Parameters:
        frequencies (dict): A dictionary of word frequencies.
        background_color (str): Background color of the word cloud.
        
    Returns:
        WordCloud: A WordCloud object ready for plotting.
    """
    wc = WordCloud(background_color=background_color)
    frequencies = {word: freq + 1e-7 for word, freq in frequencies.items()}  # Avoid zero frequencies
    wc.generate_from_frequencies(frequencies)
    return wc

def plot_word_cloud(weights_matrix, id2token, background_color='white'):
    """
    Plot word clouds for each topic based on the topic-word distribution matrix.
    
    Parameters:
        weights_matrix (numpy.ndarray): Matrix where each row represents a topic and each column a word's weight.
        id2token (dict): Mapping from word IDs to words.
        background_color (str): Background color for the word clouds.
    """
    ntopics = weights_matrix.shape[0]
    wc_texts = [get_word_cloud_text(weights_matrix[r], id2token) for r in range(ntopics)]
    clouds = [create_word_cloud(text, background_color=background_color) for text in wc_texts]

    ncols = 3
    nrows = (ntopics + ncols - 1) // ncols  # Ensure correct number of rows

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))

    for i, cloud in enumerate(clouds):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(cloud, interpolation='bilinear')
        ax.grid(False)
        ax.axis('off')
    
    # Turn off any remaining empty subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    plt.savefig(f"./pics/top_words_D{ntopics}.jpg", dpi=150)
    plt.show()



def visualize_topics(idx2word, doc_term_matrix, model_dir='./models', output_dir='./pics', ):
    """
    Generate and save interactive visualizations for LDA models stored in a directory.
    
    Parameters:
        model_dir (str): Path to the directory containing saved LDA models.
        output_dir (str): Path to the directory where the visualizations will be saved.
        doc_term_matrix (list of list of tuples): The corpus in bag-of-words format.
        idx2word (corpora.Dictionary): Gensim dictionary mapping word IDs to words.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all directories in the model directory
    for fname in os.listdir(model_dir):
        model_path = os.path.join(model_dir, fname)
        
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, 'lda_model')
            
            if os.path.exists(model_file):
                # Load the LDA model
                model = LdaModel.load(model_file)
                
                # Create a visualization using pyLDAvis
                vis = gensimvis.prepare(
                    topic_model=model,
                    corpus=doc_term_matrix,
                    dictionary=idx2word,
                    sort_topics=False  # Keep topic order of Gensim's model unchanged
                )
                
                # Save the visualization as an HTML file
                output_file = os.path.join(output_dir, f'lda_D{fname}.html')
                pyLDAvis.save_html(vis, output_file)
                print(f'Saved LDA visualization for model {fname} to {output_file}.')
            else:
                print(f'Model file not found in directory: {model_path}')
                
                
def generate_topic_distributions(ecli, doc_term_matrix, ntopics):
    """
    Generate document embeddings based on topic distributions from a trained LDA model.
    
    Parameters:
        ecli (list of str): List of document identifiers (e.g., ECLI codes).
        doc_term_matrix (list of lists of tuples): Bag-of-words representation of the documents.
        ntopics (int): Number of topics used in the LDA model.
    
    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a document and each column to a topic. 
                      The values represent the proportion of the document belonging to each topic.
    """
    # Construct the path to the LDA model based on the number of topics
    model_path = f'./models/{ntopics}/'
    model_file = os.path.join(model_path, 'lda_model')
    
    # Load the trained LDA model
    model = LdaModel.load(model_file)
    
    # Create a DataFrame with the topic distribution for each document
    df_top_distr = pd.DataFrame(
        [dict(model[doc_term_matrix[i]][0]) for i, _ in enumerate(ecli)],
        index=ecli,
        columns=list(range(ntopics))
    )
    
    # Fill any missing values (NaN) with 0
    return df_top_distr.fillna(value=0)
                



