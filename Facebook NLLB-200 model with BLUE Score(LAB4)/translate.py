from google.cloud import translate_v2 as google_translate
import sacrebleu
from datasets import load_dataset
from yandexfreetranslate import YandexFreeTranslate

# Initialize Google Translate API client
google_client = google_translate.Client()

# Initialize Yandex Free Translate
yandex_client = YandexFreeTranslate(api="ios")

def translate_with_google(text, target_language):
    # Google Translate API request
    translation = google_client.translate(text, target_language=target_language)
    return translation['translatedText']

def translate_with_yandex(text, target_language):
    # Yandex Free Translate request
    translation = yandex_client.translate('en', target_language, text)
    return translation['text'][0]  # Assuming the first translation is the most relevant

# Load the dataset
dataset = load_dataset("facebook/flores", "eng_Latn")["devtest"]  # Example for English

# Translate sentences and evaluate
for translator in [translate_with_google, translate_with_yandex]:
    translated_sentences = [translator(example['sentence'], 'ru') for example in dataset]  # Example for English to Russian
    references = [[example['sentence']] for example in dataset]
    bleu_score = sacrebleu.corpus_bleu(translated_sentences, references).score
    print(f"BLEU score using {translator.__name__}: {bleu_score}")
