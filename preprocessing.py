import spacy
from spacy.matcher import PhraseMatcher 
from spacy import Language
from spacy.tokens import Doc, Span
from spacy.lang.nl import stop_words
from spacy.util import filter_spans

from typing import List
import numpy as np


# Define common legal words and terms often found in Dutch case law related to eviction
COMMON_WORDS = [
    'heer', 'mevrouw', 'betrokkene', 'gedaagde', 'gedaagd', 'eisen', 'eiser', 'mate', 
    'maatregel', 'aldus', 'rechter', 'voorzieningenrechter', 'opposant', 'dagen',
    'recoventie', 'conventie', 'instellen', 'huurster', 'woningstichting', 
    'woonstad rotterdam', 'mitros', 'havensteder', 'ymere', 'waterweg', 'portaal', 
    'woonpunt', 'woonwaard', 'rochdale', 'daagden', 'gedaagden', 'gedaagdad', 
    'stichting', 'rechtbank', 'beroep', 'reconventie', 'eisend', 'verweerster', 
    'haard', 'eisere', 'verweren', 'onderbouwen', 'verzoeker', 'verzoek', 'verzoekster',
    'verzoekschrift', 'voorlopig', 'verweerder', 'ecli', 'rechthebben', 'rechthebbende',
]

# Combine default Dutch stop words with custom legal terms
STOP_WORDS = list(stop_words.STOP_WORDS) + COMMON_WORDS

# Load company names related to Dutch housing corporations
with np.load('Woningcorporations_names.npz') as f:
    COMPANY_NAMES = [str(n) for n in f['names']]
    

# Load the pre-trained Dutch language model
MODEL = 'nl_core_news_lg'
nlp = spacy.load(MODEL, exclude=["parser", "tagger"])


def create_phrase_matcher(phrases: List[str]) -> PhraseMatcher:
    """
    Create a PhraseMatcher object to detect specific phrases (e.g., company names) in text.
    
    Parameters:
        phrases (List[str]): A list of phrases to match.
    
    Returns:
        PhraseMatcher: A configured PhraseMatcher object.
    """
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    #patterns = nlp.pipe(phrases)
    patterns = list(nlp.pipe(phrases))
    matcher.add('COMPANYNAMES', patterns)
    return matcher

# Initialize the PhraseMatcher with company names
matcher_comp_names = create_phrase_matcher(phrases=COMPANY_NAMES)


# Define a component to detect companies' names
@Language.component("company_name_detector")
def company_name_detector(doc: Doc) -> Doc:
    """
    Spacy pipeline component to detect company names in a document and label them as 'COMPANY'.
    
    Parameters:
        doc (Doc): The document to process.
    
    Returns:
        Doc: The processed document with detected company names labeled.
    """
    matches = matcher_comp_names(doc)
    spans = [Span(doc, start, end, label="COMPANY") for match_id, start, end in matches]
    filtered_spans = filter_spans(list(doc.ents) + spans)
    doc.ents = filtered_spans
    return doc
    

# Add custom components to the Spacy pipeline
nlp.add_pipe("company_name_detector", after="ner")
nlp.add_pipe('merge_entities')

# Adjust the maximum length for processing large documents
nlp.max_length = 2000000

print(f"Here is the pipeline for the model '{MODEL}': \n\t{nlp.pipe_names}")


