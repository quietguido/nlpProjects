
First Part:
1. Install NLTK, download WordNet data.
2. Download and review SimLex999 data.
3. Calculate word similarities based on WordNet’s path_similarity (iterate over all
synsets pairs the words belong to, account for POS tags). Are any words from SimLex999
missing in WordNet?

Example: 
Word1: enter, Word2: owe, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.3333333333333333
Word1: portray, Word2: notify, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.1111111111111111
Word1: remind, Word2: sell, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.25
Word1: absorb, Word2: possess, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.3333333333333333
Word1: join, Word2: acquire, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.3333333333333333
Word1: send, Word2: attend, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.2
Word1: gather, Word2: attend, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.25
Word1: absorb, Word2: withdraw, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.3333333333333333
Word1: attend, Word2: arrive, SimLex999 POS: V, Maximum Path Similarity by WordNet: 0.25
No missing word pairs in WordNet.
Kendall's tau: 0.35344887126870356, p-value: 7.744308980342708e-55

Second part: 
4. Install fastText, download English fastText model in binary format
(https://fasttext.cc/docs/en/crawl-vectors.html).
5. Calculate word similarities based on cosine similarity of word vectors (note that e.g.
scipy.spatial.distance.cosine returns ). Report if any words are missing in
the model.
6. Calculate Kendall’s tau (e.g. using scipy.stats.kendalltau) between the gold
standard and obtained scores (use only word pairs processed by all models). Summarize findings in a table and analyze them.

Example:
Word1: bring, Word2: complain, Cosine Similarity: 0.22269386053085327
Word1: enter, Word2: owe, Cosine Similarity: 0.09995695948600769
Word1: portray, Word2: notify, Cosine Similarity: 0.04743067920207977
Word1: remind, Word2: sell, Cosine Similarity: 0.18631869554519653
Word1: absorb, Word2: possess, Cosine Similarity: 0.31259673833847046
Word1: join, Word2: acquire, Cosine Similarity: 0.2293568104505539
Word1: send, Word2: attend, Cosine Similarity: 0.3509646952152252
Word1: gather, Word2: attend, Cosine Similarity: 0.3774953782558441
Word1: absorb, Word2: withdraw, Cosine Similarity: 0.2973150610923767
Word1: attend, Word2: arrive, Cosine Similarity: 0.3870837986469269
No missing word pairs in fastText Model.
Kendall's tau: 0.3301400933912036, p-value: 7.744002627565699e-55

Overall in both models we have found all the words from SIM_LEX-999.
Similarity analysis with golden truth and models resulted in:

WordNet 
Kendall's tau: 0.35344887126870356, p-value: 7.744308980342708e-55

fastText
Kendall's tau: 0.3301400933912036, p-value: 7.744002627565699e-55

Result show quite large difference. 
It can be caused from words with similarities in meaning, but difference in usage.
