
import stanza
from seqeval.metrics import classification_report

pipeline = stanza.Pipeline(lang='kk', processors='tokenize,lemma,pos', tokenize_no_ssplit=True)

def concat_text_fields(file_path):
    concatenated_text = []
    current_text = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith('# text = '):
                current_text.append(line[len('# text = '):])
            elif not line:
                if current_text:
                    concatenated_text.append(' '.join(current_text))
                    current_text = []

    if current_text:
        concatenated_text.append(' '.join(current_text))

    return '\n\n'.join(concatenated_text)


def parse_data(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            elif line[0].isdigit():
                parts = line.strip().split('\t')

                if len(parts) > 4:
                    token_data = {'id': parts[0], 'form': parts[1], 'lemma': parts[2], 'upos': parts[3]}

                    if parts[2] != '_':
                        current_sentence.append(token_data)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences



# Function to collect lemmas and POS tags from KTB annotations
def collect_ktb_annotations(file_path):
    lemmas, pos_tags = [], []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip comments and empty lines
                continue

            columns = line.split('\t')
            lemmas.append(columns[2])  # Lemma is in the third column
            pos_tags.append(columns[3])  # POS tag is in the fourth column

    return lemmas, pos_tags


def compare_with_stanza(ktbLemmas, ktb_pos, text, language_code='your_language_code'):
    pipeline = stanza.Pipeline(lang=language_code, processors='tokenize,lemma,pos', tokenize_no_ssplit=True)
    doc = pipeline(text)

    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
    pos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]

    return lemmas, pos_tags

# Function to collect lemmas and POS tags from Stanza pipeline
def collect_stanza_text(text, language_code='your_language_code'): 
    pipeline = stanza.Pipeline(lang=language_code, processors='tokenize,lemma,pos', tokenize_no_ssplit=True)
    doc = pipeline(text)

    return doc

def collect_stanza_annotations(text, language_code='your_language_code'):
    pipeline = stanza.Pipeline(lang=language_code, processors='tokenize,lemma,pos', tokenize_no_ssplit=True)
    doc = pipeline(text)

    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
    pos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]

    return lemmas, pos_tags



language_code = 'kk'  # Kazakh language code
ktb_file_path = 'ktb/kk_ktb-ud-test.conllu'

# Collect Stanza annotations
ktb_text = concat_text_fields(ktb_file_path) 
truth_machine = parse_data(ktb_file_path)
text_stanza = collect_stanza_text(ktb_text, language_code)

count = 0
mismatched_indices = []
posMatches = 0
posCount = 0
lemmaMatches = 0
lemmaCount = 0
for i, (stanza_sentence, truth_sentance) in enumerate(zip(text_stanza.sentences, truth_machine)):
        if len(stanza_sentence.words) != len(truth_sentance):
            print(f"Wrong identified sentence index {i + 1} ")
            mismatched_indices.append(i)
            count+=1
        else:
            for i in range(len(stanza_sentence.words)):
                posCount+=1
                lemmaCount+=1
                if stanza_sentence.words[i].lemma == truth_sentance[i]['lemma']:
                    lemmaMatches+=1
                if stanza_sentence.words[i].upos == truth_sentance[i]['upos']:
                    posMatches+=1
        
lemmatization_accuracy = lemmaMatches / lemmaCount * 100
pos_accuracy = posMatches / posCount * 100

print(f"Lemmatization Accuracy: {lemmatization_accuracy:.5f}%")# arround 98%
print(f"POS Tagging Accuracy: {pos_accuracy:.5f}%")# arround 98%


# Upon thorough examination of the results, it becomes evident that a significant number of errors
# are concentrated in two main categories: words containing dashes and words with ambiguous meanings.

# 1. Words with Dashes:
# The pipeline seems to encounter challenges when processing words that include dashes. 
# This might be attributed to difficulties in tokenization and parsing of compound words.
# As a result, the model tends to misinterpret the structure of such words,
# leading to errors in lemmatization and part-of-speech tagging.

# 2. Ambiguous Word Meanings:
# Another notable source of errors is words with ambiguous meanings.
# These are instances where a single word has multiple interpretations, 
# making it challenging for the model to accurately determine the correct
# lemma and part-of-speech tag.

# 3 task 

def extract_plain_text_from_file(file_path):
    plain_text = ""
    types = []
    current = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip empty lines and lines starting with #
            if not line.strip() or line.startswith('#'):
                continue

            # Split the line into columns
            columns = line.split(' ')

            # Extract the word form (column index 1)
            word_form = columns[0]
            if(columns[1].strip() != 'O'):
                current.append(columns[1].strip())

            # Append the word form to the plain text list
            plain_text += word_form

            if word_form == '.':
                plain_text += '\n\n'
                types.append(current)
                current = []
            else: 
                plain_text += ' '

    # Join the words to create the plain text
    return plain_text, types


# Example usage
file_path = 'kaznerd/IOB2_test.txt'  # Replace with the actual path to your CoNLL-U file
plain_text_from_file, types = extract_plain_text_from_file(file_path)
print(plain_text_from_file)

def stanford_ner(text):
    # Load the Stanza pipeline with tokenizer and NER
    pipeline = stanza.Pipeline(lang='kk', processors='tokenize, ner', tokenize_pretokenized=True)
    
    # Process the text through the pipeline
    doc = pipeline(text)
    
    # Extract NER annotations
    ner_annotations = []
    for sentence in doc.sentences:
        cur = []
        for word, entity in zip(sentence.words, sentence.entities):
            cur.append(entity.type)
        ner_annotations.append(cur)

    return ner_annotations


ner_annotations_stanza = stanford_ner(plain_text_from_file)

corrected_types = []
corrected_ner_annotations_stanza = []

for i in range(len(types)):
    if len(types[i]) == len(ner_annotations_stanza[i]) and len(types[i]) != 0 : 
        corrected_types.append(types[i])
        corrected_ner_annotations_stanza.append(ner_annotations_stanza[i])

report = classification_report(corrected_types, corrected_ner_annotations_stanza)
print(report)

# ISSUE with classification_report it cannot classify types with B and I 

# ['B-TIME', 'I-TIME', 'I-TIME', 'B-CARDINAL']
# ['TIME', 'CARDINAL']
# ['B-TIME', 'I-TIME', 'I-TIME', 'B-CARDINAL']
# ['TIME', 'CARDINAL']
# ['B-POSITION', 'B-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'B-POSITION', 'I-POSITION', 'B-GPE']
# ['POSITION', 'DATE', 'POSITION', 'PROJECT']
# ['B-POSITION', 'B-DATE', 'I-DATE', 'I-DATE', 'B-POSITION', 'I-POSITION', 'B-GPE']
# ['POSITION', 'DATE', 'POSITION', 'PROJECT']
# ['B-PERSON', 'I-PERSON']
# ['PERSON']
# ['B-GPE']
# ['GPE']
# ['B-CARDINAL', 'B-CARDINAL', 'B-CARDINAL', 'I-CARDINAL', 'I-CARDINAL', 'I-CARDINAL']
# ['CARDINAL', 'CARDINAL', 'CARDINAL']
# ['B-CARDINAL', 'B-CARDINAL', 'B-CARDINAL', 'I-CARDINAL', 'I-CARDINAL', 'I-CARDINAL', 'I-CARDINAL']
# ['CARDINAL', 'CARDINAL', 'CARDINAL']
# ['B-CARDINAL']
# ['CARDINAL']
# ['B-CARDINAL']
# ['CARDINAL']
# ['B-ORGANISATION']
# ['ORGANISATION']