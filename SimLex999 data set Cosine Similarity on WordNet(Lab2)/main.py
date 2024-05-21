import nltk
nltk.download('wordnet')
# nltk.download('omw-1.4')
from scipy.stats import kendalltau
import pandas as pd

# Load the SimLex999 data
simlex999_path = 'simlex/simLex-999.txt' 
simlex999_data = pd.read_csv(simlex999_path, sep="\t")

# print(simlex999_data.head())

from nltk.corpus import wordnet as wn

'''
First Part:
1. Install NLTK, download WordNet data.
2. Download and review SimLex999 data.
3. Calculate word similarities based on WordNet’s path_similarity (iterate over all
synsets pairs the words belong to, account for POS tags). Are any words from SimLex999
missing in WordNet?
'''

def get_wordnet_pos(simlex_pos):
    """Map SimLex999 POS tags to WordNet POS tags."""
    if simlex_pos == 'A':  # Adjective
        return wn.ADJ
    elif simlex_pos == 'N':  # Noun
        return wn.NOUN
    elif simlex_pos == 'V':  # Verb
        return wn.VERB
    elif simlex_pos == 'R':  # Adverb
        return wn.ADV
    else:
        return None

def calculate_wordnet_similarity(word1, word2, pos):
    pos = get_wordnet_pos(pos)  # Convert SimLex POS to WordNet POS
    similarities = []

    synsets1 = wn.synsets(word1, pos=pos)
    synsets2 = wn.synsets(word2, pos=pos)

    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None:
                similarities.append(similarity)
    
    return max(similarities) if similarities else None

missing_words = []
golden_truth_array = []
wordnet_similarity_array = []

print('Start: ')
for index, row in simlex999_data.iterrows():
    word1, word2, pos, golden_truth = row['word1'], row['word2'], row['POS'], row['SimLex999']
    similarity = calculate_wordnet_similarity(word1, word2, pos)

    if similarity is None:
        missing_words.append((word1, word2))
        continue

    golden_truth_array.append(golden_truth)
    wordnet_similarity_array.append(similarity)

    print(f"Word1: {word1}, Word2: {word2}, SimLex999 POS: {pos}, Maximum Path Similarity by WordNet: {similarity}")

if missing_words:
    print("Missing word pairs in WordNet:", missing_words)
else:
    print("No missing word pairs in WordNet.")

tau, p_value = kendalltau(golden_truth_array, wordnet_similarity_array)
print(f"Kendall's tau: {tau}, p-value: {p_value}")

'''
No missing word pairs in WordNet.
Kendall's tau: 0.35344887126870356, p-value: 7.744308980342708e-55
'''

'''
Second part: 
4. Install fastText, download English fastText model in binary format
(https://fasttext.cc/docs/en/crawl-vectors.html).
5. Calculate word similarities based on cosine similarity of word vectors (note that e.g.
scipy.spatial.distance.cosine returns ). Report if any words are missing in
the model.
6. Calculate Kendall’s tau (e.g. using scipy.stats.kendalltau) between the gold
standard and obtained scores (use only word pairs processed by all models). Summarize findings in a table and analyze them.

'''
import fasttext

model = fasttext.load_model('cc.en.300.bin')

from scipy.spatial.distance import cosine

def get_cosine_similarity(model, word1, word2):
    if word1 not in model.words or word2 not in model.words:
        return None
    
    vector1 = model.get_word_vector(word1)
    vector2 = model.get_word_vector(word2)

    cos_sim = 1 - cosine(vector1, vector2)
    return cos_sim

print('Start Second: ')
golden_truth_array = []
cosine_similarity_array = []
missing_words.clear()

for index, row in simlex999_data.iterrows():
    word1, word2, golden_truth = row['word1'], row['word2'], row['SimLex999']
    cosine_similarity = get_cosine_similarity(model, word1, word2)

    if cosine_similarity is None:
        missing_words.append((word1, word2))
        continue

    golden_truth_array.append(golden_truth)
    cosine_similarity_array.append(cosine_similarity)

    print(f"Word1: {word1}, Word2: {word2}, Cosine Similarity: {cosine_similarity}")


if missing_words:
    print("Missing word pairs in fastText Model:", missing_words)
else:
    print("No missing word pairs in fastText Model.")

from scipy.stats import kendalltau

tau, p_value = kendalltau(golden_truth_array, cosine_similarity_array)
print(f"Kendall's tau: {tau}, p-value: {p_value}")

'''
No missing word pairs in fastText Model.
Kendall's tau: 0.3301400933912036, p-value: 7.744002627565699e-55
'''