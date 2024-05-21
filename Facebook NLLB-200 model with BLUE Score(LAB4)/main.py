# 1. Read the HF documentation. Find HF translation models that support English-Kazakh
# (alternatively English—Russian) translation.
# 2. Start with Facebook’s NLLB model https://huggingface.co/facebook/nllb-200-distilled-600M
# Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., ... & NLLB
# Team. (2022). No language left behind: Scaling human-centered machine translation. arXiv
# preprint arXiv:2207.04672.
# 3. Read the description of the BLEU score https://en.wikipedia.org/wiki/BLEU and its
# implementation https://github.com/mjpost/sacrebleu
# 4. Evaluate the NLLB model using the devtest subset of the FLORES dataset (evaluate both
# directions: En—Xx and Xx—En). (20)
# 5. Collect a small (~50 sentences) parallel corpus for your language pair. Use two different
# genres, e.g. news and fiction. Align the data; evaluate the NNLB model on the dataset. (20)
# 6. Evaluate another model from the HF on two datasets (FOLORES and your own). (20)
# 7. Evaluate Google Translate and Yandex Translate on the same datasets. (20)
# 8. Summarize and analyze the evaluation results (BLEU scores). Manually inspect sentence pairs
# with lowest/highest scores and provide analysis. (20)

'''
map 
Huggingface NLLB model 
dataset FLORES
eng -> rus / rus -> eng
bleu score to measure the performance (input text and translated text)

'''



from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import sacrebleu
import torch
import numpy as np

# Load the model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to('cuda')  # Move model to GPU if available

def test_on_model(lang: str):

  # Load the FLORES dataset
  flores_dataset = load_dataset("facebook/flores", lang)
  print(flores_dataset["devtest"])

  # Function to translate text in batches
  def translate_in_batches(texts, model, tokenizer, batch_size=10):
      translated_texts = []
      for i in range(0, len(texts), batch_size):
          batch = texts[i:i+batch_size]
          inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
          with torch.no_grad():
              outputs = model.generate(**inputs)
          batch_translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
          translated_texts.extend(batch_translated_texts)
      return translated_texts

  # Evaluate the model in batches
  texts = [example["sentence"] for example in flores_dataset["devtest"]]
  translated_texts = translate_in_batches(texts, model, tokenizer, batch_size=1)

  # Calculate BLEU scores for each translation
  bleu_scores = [sacrebleu.raw_corpus_bleu([t], [[o]]).score for t, o in zip(translated_texts, texts)]

  # Calculate average BLEU score
  average_bleu_score = np.mean(bleu_scores)
  print(f"Average BLEU score: {average_bleu_score}")

  # Find the best and worst 10 BLEU scores
  sorted_scores_indices = np.argsort(bleu_scores)
  best_10_indices = sorted_scores_indices[-10:]
  worst_10_indices = sorted_scores_indices[:10]

  best_10_scores = [(texts[i], translated_texts[i], bleu_scores[i]) for i in best_10_indices]
  worst_10_scores = [(texts[i], translated_texts[i], bleu_scores[i]) for i in worst_10_indices]

  print("\nBest 10 BLEU Scores:")
  for original, translated, score in best_10_scores:
      print(f"BLEU score: {score}, Original: {original[:50]}, Translated: {translated[:50]}")

  print("\nWorst 10 BLEU Scores:")
  for original, translated, score in worst_10_scores:
      print(f"BLEU score: {score}, Original: {original[:50]}, Translated: {translated[:50]}")

test_on_model("rus_Cyrl")
test_on_model("eng_Latn")

'''
Average BLEU score: 9.184349219470679

Best 10 BLEU Scores:
BLEU score: 100.00000000000004, Original: По словам Мика О'Флинна, исполняющего обязанности , Translated: По словам Мика О'Флинна, исполняющего обязанности 
BLEU score: 100.00000000000004, Original: NHK также сообщила, что АЭС Касивазаки Карива, рас, Translated: NHK также сообщила, что АЭС Касивазаки Карива, рас
BLEU score: 100.00000000000004, Original: Полиция штата Мадхья-Прадеш вернула украденные ноу, Translated: Полиция штата Мадхья-Прадеш вернула украденные ноу
BLEU score: 100.00000000000004, Original: Рок-группа должна была гастролировать по США и Кан, Translated: Рок-группа должна была гастролировать по США и Кан
BLEU score: 100.00000000000004, Original: Он получил образование акушера-гинеколога и начал , Translated: Он получил образование акушера-гинеколога и начал 
BLEU score: 100.00000000000004, Original: Созерцание цветения сакуры, называемое "ханами", в, Translated: Созерцание цветения сакуры, называемое "ханами", в
BLEU score: 100.00000000000004, Original: Фотограф был доставлен в медицинский центр имени Р, Translated: Фотограф был доставлен в медицинский центр имени Р
BLEU score: 100.00000000000004, Original: Большинство отдельных балтийских круизов включают , Translated: Большинство отдельных балтийских круизов включают 
BLEU score: 100.00000000000004, Original: Произношение в итальянском языке относительно прос, Translated: Произношение в итальянском языке относительно прос
BLEU score: 100.00000000000004, Original: В 1960-х гг. Бжезинский занимал должность советник, Translated: В 1960-х гг. Бжезинский занимал должность советник

Worst 10 BLEU Scores:
BLEU score: 0.0, Original: Если вы обнаружите, что переустанавливаете будильн, Translated: Dacă afli că reinstalaţi zgomotul în somn, îl pute
BLEU score: 0.0, Original: Тибетский буддизм основан на учениях Будды, но был, Translated: Το Θιβετιανό Βουδισμό βασίστηκε στις διδασκαλίες τ
BLEU score: 0.0, Original: Принцип тибетского буддизма очень прост. Он состои, Translated: مبدأ البوذية التبتية بسيط جدا. يتكون من اليوغا وال
BLEU score: 0.0, Original: Энергия кундалини (энергия просветления) в кундали, Translated: कुंडलिनी-योग में ऊर्जा योग, श्वसन व्यायाम, मंत्र औ
BLEU score: 0.0, Original: Сердцем тибетской медитации считается Йога Божеств, Translated: तिब्बती ध्यान का केंद्र योग देवता माना जाता है। वि
BLEU score: 0.0, Original: В саванне приматам с такой пищеварительной системо, Translated: Dans un système digestif comme celui de l'homme, i
BLEU score: 0.0, Original: Более того, невыполнение этого приведёт к таким се, Translated: De plus, le non-respect de ces règles entraînera d
BLEU score: 0.0, Original: Наиболее доступными растительными ресурсами были б, Translated: Les plus accessibles ressources végétales seraient
BLEU score: 0.0, Original: Напротив, еда животного происхождения (муравьи, те, Translated: En revanche, la nourriture d'origine animale (mura
BLEU score: 0.0, Original: Принимая во внимание все вышесказанное, не стоит у, Translated: In die lig van al die bogenoemde, is dit nie verba



Average BLEU score: 0.7722331390226777

Best 10 BLEU Scores:
BLEU score: 8.854416306086701, Original: During the struggle for independence organised by , Translated: Pendant la lutte pour l'indépendance organisée par
BLEU score: 9.092735920162841, Original: The Giza Plateau, or "Giza Necropolis" in the Egyp, Translated: တီႈၼိူဝ်လိၼ် Giza Plateau ဢမ်ႇၼၼ် "Giza Necropolis
BLEU score: 9.14462689237287, Original: Scotturb Bus 403 travels regularly to Sintra, stop, Translated: L'autobus 403 di Scotturb viaggia regolarmente a S
BLEU score: 9.53295972797249, Original: Their thermal behavior is not as steady as large c, Translated: Hulle termiese gedrag is nie so stabiel soos groot
BLEU score: 10.640850690356462, Original: Hokuriku Electric Power Co. reported no effects fr, Translated: Hokuriku Electric Power Co. informó que no hubo ef
BLEU score: 10.727295782787309, Original: Current senator and Argentine First Lady Cristina , Translated: Nanambara ny kandidà filoham-pirenena azy ny senat
BLEU score: 12.673718536830808, Original: For example, “learning” and “socialization” are su, Translated: Zum Beispiel werden learning und socialization als
BLEU score: 12.673718536830808, Original: During the 1980s he worked on shows such as Taxi, , Translated: Au cours des années 1980, il a travaillé sur des é
BLEU score: 20.105373454060025, Original: The photographer was transported to Ronald Reagan , Translated: Le photographe a été transporté au Ronald Reagan U
BLEU score: 33.16186519505936, Original: Two popular content theories are Maslow's Hierarch, Translated: သွင်ၶေႃႈႁၼ်ထိုင်ဢၼ်ၸိုဝ်ႈသဵင်ယႂ်ႇၶွင်လွင်ႈၵမ်ႉထႅမ်

Worst 10 BLEU Scores:
BLEU score: 0.0, Original: Such things have become separate disciplines, whic, Translated: လွင်ႈၸိူဝ်းၼႆႉ ပဵၼ်မႃး လွင်ႈပၵ်းပိူင်ဢၼ်ၽၢတ်ႇဢွၵ်ႇ
BLEU score: 0.0, Original: The mob of people forced the King And Queen to hav, Translated: Multimea de oameni i-a forţat pe rege şi regină să
BLEU score: 0.0, Original: At one point a member of the mob waved the head of, Translated: Σε ένα σημείο, ένα μέλος του πλήθους κουνιόταν το 
BLEU score: 0.0, Original: The war expenditures of U.S. imperialism in the co, Translated: As despesas de guerra do imperialismo americano na
BLEU score: 0.0, Original: Of course, the superprofits derived from the protr, Translated: Φυσικά, τα υπερκεράσματα που προέρχονται από την π
BLEU score: 0.0, Original: To understand the Templars one must understand the, Translated: Para entender a los Templarios uno debe entender e
BLEU score: 0.0, Original: Up means you should start at the tip and push the , Translated: arriba significa que debes comenzar en la punta y 
BLEU score: 0.0, Original: All these things and more highlight Ontario as wha, Translated: Toutes ces choses et bien d'autres mettent en évid
BLEU score: 0.0, Original: Large areas further north are quite sparsely popul, Translated: De vastes régions plus au nord sont assez peu peup
BLEU score: 0.0, Original: Examples include control, planning and scheduling,, Translated: Beispiele sind Kontrolle, Planung und Planung, die
'''


from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import sacrebleu

# Load the tokenizer and model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def test_on_model(src_lang: str, tgt_lang: str):
    
    # Load the FLORES dataset
    flores_dataset = load_dataset("facebook/flores", src_lang)
    print(flores_dataset["devtest"])

    # Move model to GPU if available
    device = 0 if torch.cuda.is_available() else -1
    translation_pipeline_en_to_ru = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, device=device)

    # Function to translate text in batches
    def translate(texts, translation_pipeline):
      translated_texts = []
      batch_size = 16  # Adjust based on your GPU memory
      
      # Ensure the pipeline and tensors are moved to the GPU
      if torch.cuda.is_available():
          translation_pipeline.model.to('cuda')

      for i in range(0, len(texts), batch_size):
          batch = texts[i:i + batch_size]
          
          # Directly use the pipeline without manual tensor conversion
          # The pipeline handles moving data to the GPU internally
          with torch.no_grad():  # Ensure no gradients are calculated
              translated_batch = translation_pipeline(batch)
          
          # Extract translation texts
          batch_translations = [t['translation_text'] for t in translated_batch]
          translated_texts.extend(batch_translations)

      return translated_texts

    texts = [example["sentence"] for example in flores_dataset["devtest"]]
    translated_texts = translate(texts, translation_pipeline_en_to_ru)
    
    original_sentences = texts
    translated_sentences = translated_texts

    # Calculate the BLEU score
    bleu_score = sacrebleu.corpus_bleu(translated_sentences, [original_sentences], lowercase=True).score

    print("BLEU score:", bleu_score)

# Call the function with the desired source and target languages
test_on_model("rus_Cyrl", "eng_Latn")

#test_on_model("rus_Cyrl", "eng_Latn")
#BLEU score: 1.1066054642980447
#for batch of 8


#BLEU score: 1.1066054642980447
#for batch of 16

test_on_model("eng_Latn", "rus_Cyrl")


#BLEU score: 1.1727149651056463
#fror batch of 8


english_text = '''
The two-day conference, which began on Sunday, was the second of its kind. It was held to try to chart a course forward for international engagement with the country. But the Taliban administration took issue with the inclusion of some groups at the meeting. Attended by special envoys from 25 countries and regional organizations, the conference is aimed at increasing international engagement with Afghanistan and developing a more coordinated response to the problems afflicting the war-torn nation.

The Taliban administration, the de facto rulers of Afghanistan since 2021, had been invited to the conference but at the last minute the group said it would not attend. In a statement, the Taliban’s Ministry of Foreign Affairs said it should be the sole official representative of Afghanistan for talks with the international community and only then could engage in frank discussions. Inclusion of others would hinder progress, the statement added.

“This government of Afghanistan cannot be coerced by anyone,” it stated.

Representatives from Afghan civil society, women’s groups, the Organization of Islamic Cooperation, the European Union and the Shanghai Cooperation Organization were present at the conference. Afghan political opposition parties, including the National Resistance Front, which has a small armed wing, were not invited, although they had asked to be included.

The Taliban administration’s decision, announced on the eve of the conference, appeared to have been made to avoid awkward conversations with Afghans living outside the country who oppose the Taliban authorities’ exclusion of women, and political opponents inside Afghanistan, several delegates said.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
'''


import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sacrebleu

nltk.download('punkt')

# Load the tokenizer and model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def test_on_plain_text(plain_text: str, src_lang: str, tgt_lang: str):
    # Split the plain text into sentences
    sentences = nltk.tokenize.sent_tokenize(plain_text)

    # Move model to GPU if available
    device = 0 if torch.cuda.is_available() else -1
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, device=device)

    # Function to translate text in batches
    def translate(texts, translation_pipeline):
        translated_texts = []
        batch_size = 1 # Adjust based on your GPU memory
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Translate the batch
            with torch.no_grad():  # Ensure no gradients are calculated
                translated_batch = translation_pipeline(batch)
            
            # Extract translation texts
            batch_translations = [t['translation_text'] for t in translated_batch]
            translated_texts.extend(batch_translations)

        return translated_texts

    translated_texts = translate(sentences, translation_pipeline)
    
    # Calculate the BLEU score
    bleu_score = sacrebleu.corpus_bleu(translated_texts, [sentences], lowercase=True).score

    print("BLEU score:", bleu_score)

# Example usage with some plain text
english_text = '''
The two-day conference, which began on Sunday, was the second of its kind. It was held to try to chart a course forward for international engagement with the country. But the Taliban administration took issue with the inclusion of some groups at the meeting. Attended by special envoys from 25 countries and regional organizations, the conference is aimed at increasing international engagement with Afghanistan and developing a more coordinated response to the problems afflicting the war-torn nation.

The Taliban administration, the de facto rulers of Afghanistan since 2021, had been invited to the conference but at the last minute the group said it would not attend. In a statement, the Taliban’s Ministry of Foreign Affairs said it should be the sole official representative of Afghanistan for talks with the international community and only then could engage in frank discussions. Inclusion of others would hinder progress, the statement added.

“This government of Afghanistan cannot be coerced by anyone,” it stated.

Representatives from Afghan civil society, women’s groups, the Organization of Islamic Cooperation, the European Union and the Shanghai Cooperation Organization were present at the conference. Afghan political opposition parties, including the National Resistance Front, which has a small armed wing, were not invited, although they had asked to be included.

The Taliban administration’s decision, announced on the eve of the conference, appeared to have been made to avoid awkward conversations with Afghans living outside the country who oppose the Taliban authorities’ exclusion of women, and political opponents inside Afghanistan, several delegates said.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
'''

test_on_plain_text(english_text, "eng_Latn", "rus_Cyrl")
#BLEU score: 0.14045551207209941


russian_text = '''
В Челябинске козырек подъезда жилого дома обрушился и заблокировал жильцам выход из здания. О случившемся в российском городе сообщила поисково-спасательная служба Челябинской области в своем Telegram-канале.

По информации пресс-службы, инцидент произошел в 3 часа ночи во вторник, 20 февраля, в пятиэтажном доме на улице Комаровского — бетонный козырек над одним из его подъездов рухнул и полностью заблокировал входную дверь. Ситуация осложнялась тем, что в непосредственной близости от упавшего козырька находилась труба газопровода, что было чревато угрозой ее нарушения.

Работы по ликвидации последствий ЧП проводились в несколько этапов. Сначала сотрудники газовой службы перекрыли газопровод, а затем спасатели, используя альпинистское снаряжение, достигли места, в котором проломился бетон. Позже они, зависнув в подвесных системах, с помощью гидравлического аварийно-спасательного инструмента перекусили прутья арматуры и деблокировали бетонную плиту. «Козырек транспортировали от подъезда совместно с сотрудниками пожарной охраны и уже к 5 утра вход в подъезд был освобожден», — рассказали в пресс-службе. Там добавили, что в результате инцидента никто не пострадал.
У пилота Максима Кузьминова, убитого в испанском городе Вильяхойоса в Аликанте, обнаружили поддельные документы. Об этом сообщает агентство EFE.

Осведомленные с ходом расследования источники рассказали, что обнаруженный рядом с телом паспорт был выписан на 33-летнего гражданина Украины. Однако пилоту-перебежчику было лишь 28 лет.

Ранее EFE подтвердило, что убитым на парковке в Вильяхойосе оказался именно Кузьминов. Криминалисты обнаружили в его теле больше пяти огнестрельных ран.

Накануне отец Никиты Кирьянова, который был техником и служил в одном экипаже с Кузьминовым, отреагировал на сообщения о смерти пилота. Он заявил, что «хочет дождаться официального подтверждения от властей о его ликвидации».
Венесуэла в скором времени вступит в состав БРИКС. Об этом заявил президент страны Николас Мадуро, сообщает ТАСС.

«Тенденция возникновения нового мира и консолидации БРИКС необратима, и Венесуэла скоро войдет в БРИКС», — подчеркнул он.

По словам лидера республики, новый мир с БРИКС уже существует и необратим. Мадуро отметил, что он пришел на смену «старому колониальному миру с войнами, интервенциями, геноцидом и комплексом превосходства».

Ранее президент Венесуэлы заявил, что Россия побеждает Запад в развязанном им же конфликте на Украине. Он уточнил, что «эта война оказалась невыносима для США».

Губернатор Белгородской области Вячеслав Гладков предупредил о мошенниках, которые подделывают его голос и пишут сообщения от его имени. Об этом он написал на своей странице во «ВКонтакте».

«Друзья, у меня нет такой практики с рассылкой голосовых сообщений, я этого никогда не делаю. Внимательно смотрите, с каких номеров вам пишут, приходят голосовые сообщения», — написал глава региона.

Гладков посоветовал жителям обращаться в приемную в случае, если на их телефоны приходят сообщения от имени губернатора. Помощники прояснят ситуацию и подтвердят и опровергнут те запросы, которые могут поступать, добавил он.

Ранее губернатор Красноярского края Михаил Котюков рассказал, что его пытались обмануть мошенники. Ему позвонили с незнакомого номера. Неизвестный мужчина обратился к губернатору по имени-отчеству и представился сотрудником сотового оператора. Он сообщил, что срок действия сим-карты Котюкова якобы заканчивается, после этого глава региона бросил трубку.
В конце ноября, в оттепель, часов в девять утра, поезд Петербургско-Варшавской железной дороги на всех парах подходил к Петербургу. Было так сыро и туманно, что насилу рассвело; в десяти шагах, вправо и влево от дороги, трудно было разглядеть хоть что-нибудь из окон вагона. Из пассажиров были и возвращавшиеся из-за границы; но более были наполнены отделения для третьего класса, и всё людом мелким и деловым, не из очень далека. Все, как водится, устали, у всех отяжелели за ночь глаза, все назяблись, все лица были бледно-желтые, под цвет тумана.

В одном из вагонов третьего класса, с рассвета, очутились друг против друга, у самого окна, два пассажира, – оба люди молодые, оба почти налегке, оба не щегольски одетые, оба с довольно замечательными физиономиями, и оба пожелавшие, наконец, войти друг с другом в разговор. Если б они оба знали один про другого, чем они особенно в эту минуту замечательны, то, конечно, подивились бы, что случай так странно посадил их друг против друга в третьеклассном вагоне петербургско-варшавского поезда. Один из них был небольшого роста, лет двадцати семи, курчавый и почти черноволосый, с серыми, маленькими, но огненными глазами. Нос его был широк и сплюснут, лицо скулистое; тонкие губы беспрерывно складывались в какую-то наглую, насмешливую и даже злую улыбку; но лоб его был высок и хорошо сформирован и скрашивал неблагородно развитую нижнюю часть лица. Особенно приметна была в этом лице его мертвая бледность, придававшая всей физиономии молодого человека изможденный вид, несмотря на довольно крепкое сложение, и вместе с тем что-то страстное, до страдания, не гармонировавшее с нахальною и грубою улыбкой и с резким, самодовольным его взглядом. Он был тепло одет, в широкий, мерлушечий, черный, крытый тулуп, и за ночь не зяб, тогда как сосед его принужден был вынести на своей издрогшей спине всю сладость сырой ноябрьской русской ночи, к которой, очевидно, был не приготовлен. На нем был довольно широкий и толстый плащ без рукавов и с огромным капюшоном, точь-в-точь как употребляют часто дорожные, по зимам, где-нибудь далеко за границей, в Швейцарии, или, например, в Северной Италии, не рассчитывая, конечно, при этом и на такие концы по дороге, как от Эйдткунена до Петербурга. Но что годилось и вполне удовлетворяло в Италии, то оказалось не совсем пригодным в России. Обладатель плаща с капюшоном был молодой человек, тоже лет двадцати шести или двадцати семи, роста немного повыше среднего, очень белокур, густоволос, со впалыми щеками и с легонькою, востренькою, почти совершенно белою бородкой. Глаза его были большие, голубые и пристальные; во взгляде их было что-то тихое, но тяжелое, что-то полное того странного выражения, по которому некоторые угадывают с первого взгляда в субъекте падучую болезнь. Лицо молодого человека было, впрочем, приятное, тонкое и сухое, но бесцветное, а теперь даже досиня иззябшее. В руках его болтался тощий узелок из старого, полинялого фуляра, заключавший, кажется, все его дорожное достояние. На ногах его были толстоподошвенные башмаки с штиблетами, – всё не по-русски. Черноволосый сосед в крытом тулупе все это разглядел, частию от нечего делать, и, наконец, спросил с тою неделикатною усмешкой, в которой так бесцеремонно и небрежно выражается иногда людское удовольствие при неудачах ближнего:

– Зябко?

И повел плечами.

– Очень, – ответил сосед с чрезвычайною готовностью, – и заметьте, это еще оттепель. Что ж, если бы мороз? Я даже не думал, что у нас так холодно. Отвык.

– Из-за границы, что ль?

– Да, из Швейцарии.

– Фью! Эк ведь вас!..

Черноволосый присвистнул и захохотал.

Завязался разговор. Готовность белокурого молодого человека в швейцарском плаще отвечать на все вопросы своего черномазого соседа была удивительная и без всякого подозрения совершенной небрежности, неуместности и праздности иных вопросов. Отвечая, он объявил, между прочим, что действительно долго не был в России, с лишком четыре года, что отправлен был за границу по болезни, по какой-то странной нервной болезни, вроде падучей или Виттовой пляски, каких-то дрожаний и судорог. Слушая его, черномазый несколько раз усмехался; особенно засмеялся он, когда на вопрос: «что же, вылечили?» – белокурый отвечал, что «нет, не вылечили».

– Хе! Денег что, должно быть, даром переплатили, а мы-то им здесь верим, – язвительно заметил черномазый.

– Истинная правда! – ввязался в разговор один сидевший рядом и дурно одетый господин, нечто вроде закорузлого в подьячестве чиновника, лет сорока, сильного сложения, с красным носом и угреватым лицом, – истинная правда-с, только все русские силы даром к себе переводят!

– О, как вы в моем случае ошибаетесь, – подхватил швейцарский пациент, тихим и примиряющим голосом, – конечно, я спорить не могу, потому что всего не знаю, но мой доктор мне из своих последних еще на дорогу сюда дал, да два почти года там на свой счет содержал.

– Что ж, некому платить, что ли, было? – спросил черномазый.

– Да, господин Павлищев, который меня там содержал, два года назад помер; я писал потом сюда генеральше Епанчиной, моей дальней родственнице, но ответа не получил. Так с тем и приехал.

– Куда же приехали-то?

– То есть где остановлюсь?.. Да не знаю еще, право… так…

– Не решились еще?

И оба слушателя снова захохотали.

– И небось в этом узелке вся ваша суть заключается? – спросил черномазый.

– Об заклад готов биться, что так, – подхватил с чрезвычайно довольным видом красноносый чиновник, – и что дальнейшей поклажи в багажных вагонах не имеется, хотя бедность и не порок, чего опять-таки нельзя не заметить.

Оказалось, что и это было так: белокурый молодой человек тотчас же и с необыкновенною поспешностью в этом признался.

– Узелок ваш все-таки имеет некоторое значение, – продолжал чиновник, когда нахохотались досыта (замечательно, что и сам обладатель узелка начал наконец смеяться, глядя на них, что увеличило их веселость), – и хотя можно побиться, что в нем не заключается золотых, заграничных свертков с наполеондорами и фридрихсдорами, ниже с голландскими арапчиками, о чем можно еще заключить, хотя бы только по штиблетам, облекающим иностранные башмаки ваши, но… если к вашему узелку прибавить в придачу такую будто бы родственницу, как, примерно, генеральша Епанчина, то и узелок примет некоторое иное значение, разумеется, в том только случае, если генеральша Епанчина вам действительно родственница, и вы не ошибаетесь, по рассеянности… что очень и очень свойственно человеку, ну хоть… от излишка воображения.

'''

test_on_plain_text(russian_text, "rus_Cyrl", "eng_Latn")

#BLEU score: 0.17505992174628696


import nltk
from transformers import pipeline
import sacrebleu
import torch

nltk.download('punkt')

def translate_and_evaluate(plain_text: str):
    # Split the plain text into sentences
    sentences = nltk.tokenize.sent_tokenize(plain_text)

    # Load the translation pipeline for English to Russian
    translation_pipeline = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru", device=0 if torch.cuda.is_available() else -1)

    # Translate sentences in batches
    batch_size = 8  # Adjust based on your GPU memory
    translated_texts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        translated_batch = translation_pipeline(batch)
        translated_texts.extend([t['translation_text'] for t in translated_batch])

    # Calculate the BLEU score
    bleu_score = sacrebleu.corpus_bleu(translated_texts, [sentences]).score
    print("BLEU score:", bleu_score)

# Example usage with some plain text
plain_text = '''
The two-day conference, which began on Sunday, was the second of its kind. It was held to try to chart a course forward for international engagement with the country. But the Taliban administration took issue with the inclusion of some groups at the meeting. Attended by special envoys from 25 countries and regional organizations, the conference is aimed at increasing international engagement with Afghanistan and developing a more coordinated response to the problems afflicting the war-torn nation.

The Taliban administration, the de facto rulers of Afghanistan since 2021, had been invited to the conference but at the last minute the group said it would not attend. In a statement, the Taliban’s Ministry of Foreign Affairs said it should be the sole official representative of Afghanistan for talks with the international community and only then could engage in frank discussions. Inclusion of others would hinder progress, the statement added.

“This government of Afghanistan cannot be coerced by anyone,” it stated.

Representatives from Afghan civil society, women’s groups, the Organization of Islamic Cooperation, the European Union and the Shanghai Cooperation Organization were present at the conference. Afghan political opposition parties, including the National Resistance Front, which has a small armed wing, were not invited, although they had asked to be included.

The Taliban administration’s decision, announced on the eve of the conference, appeared to have been made to avoid awkward conversations with Afghans living outside the country who oppose the Taliban authorities’ exclusion of women, and political opponents inside Afghanistan, several delegates said.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
Everyone agreed that it was worth more than all the wealth of the kingdom: but the king said, 'One feather is of no use to me, I must have the whole bird.' Then the gardener's eldest son set out and thought to find the golden bird very easily; and when he had gone but a little way, he came to a wood, and by the side of the wood he saw a fox sitting; so he took his bow and made ready to shoot at it. Then the fox said, 'Do not shoot me, for I will give you good counsel; I know what your business is, and that you want to find the golden bird. You will reach a village in the evening; and when you get there, you will see two inns opposite to each other, one of which is very pleasant and beautiful to look at: go not in there, but rest for the night in the other, though it may appear to you to be very poor and mean.' But the son thought to himself, 'What can such a beast as this know about the matter?' So he shot his arrow at the fox; but he missed it, and it set up its tail above its back and ran into the wood. Then he went his way, and in the evening came to the village where the two inns were; and in one of these were people singing, and dancing, and feasting; but the other looked very dirty, and poor. 'I should be very silly,' said he, 'if I went to that shabby house, and left this charming place'; so he went into the smart house, and ate and drank at his ease, and forgot the bird, and his country too. Time passed on; and as the eldest son did not come back, and no tidings were heard of him, the second son set out, and the same thing happened to him. He met the fox, who gave him the good advice: but when he came to the two inns, his eldest brother was standing at the window where the merrymaking was, and called to him to come in; and he could not withstand the temptation, but went in, and forgot the golden bird and his country in the same manner. Time passed on again, and the youngest son too wished to set out into the wide world to seek for the golden bird; but his father would not listen to it for a long while, for he was very fond of his son, and was afraid that some ill luck might happen to him also, and prevent his coming back. 
'''

translate_and_evaluate(plain_text)


#BLEU score: 0.10574993974397515



# pip install transformers nltk sacrebleu
import nltk
from transformers import pipeline
import sacrebleu
import torch

nltk.download('punkt')

def translate_and_evaluate_rus_to_eng(plain_text: str):
    # Split the plain Russian text into sentences
    sentences = nltk.tokenize.sent_tokenize(plain_text, language='russian')

    # Load the translation pipeline for Russian to English
    translation_pipeline = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en", device=0 if torch.cuda.is_available() else -1)

    # Translate sentences in batches
    batch_size = 8  # Adjust based on your GPU memory
    translated_texts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        translated_batch = translation_pipeline(batch)
        translated_texts.extend([t['translation_text'] for t in translated_batch])

    # Calculate the BLEU score
    bleu_score = sacrebleu.corpus_bleu(translated_texts, [sentences]).score
    print("BLEU score:", bleu_score)

# Example usage with some plain Russian text
plain_text_rus =  '''
В Челябинске козырек подъезда жилого дома обрушился и заблокировал жильцам выход из здания. О случившемся в российском городе сообщила поисково-спасательная служба Челябинской области в своем Telegram-канале.

По информации пресс-службы, инцидент произошел в 3 часа ночи во вторник, 20 февраля, в пятиэтажном доме на улице Комаровского — бетонный козырек над одним из его подъездов рухнул и полностью заблокировал входную дверь. Ситуация осложнялась тем, что в непосредственной близости от упавшего козырька находилась труба газопровода, что было чревато угрозой ее нарушения.

Работы по ликвидации последствий ЧП проводились в несколько этапов. Сначала сотрудники газовой службы перекрыли газопровод, а затем спасатели, используя альпинистское снаряжение, достигли места, в котором проломился бетон. Позже они, зависнув в подвесных системах, с помощью гидравлического аварийно-спасательного инструмента перекусили прутья арматуры и деблокировали бетонную плиту. «Козырек транспортировали от подъезда совместно с сотрудниками пожарной охраны и уже к 5 утра вход в подъезд был освобожден», — рассказали в пресс-службе. Там добавили, что в результате инцидента никто не пострадал.
У пилота Максима Кузьминова, убитого в испанском городе Вильяхойоса в Аликанте, обнаружили поддельные документы. Об этом сообщает агентство EFE.

Осведомленные с ходом расследования источники рассказали, что обнаруженный рядом с телом паспорт был выписан на 33-летнего гражданина Украины. Однако пилоту-перебежчику было лишь 28 лет.

Ранее EFE подтвердило, что убитым на парковке в Вильяхойосе оказался именно Кузьминов. Криминалисты обнаружили в его теле больше пяти огнестрельных ран.

Накануне отец Никиты Кирьянова, который был техником и служил в одном экипаже с Кузьминовым, отреагировал на сообщения о смерти пилота. Он заявил, что «хочет дождаться официального подтверждения от властей о его ликвидации».
Венесуэла в скором времени вступит в состав БРИКС. Об этом заявил президент страны Николас Мадуро, сообщает ТАСС.

«Тенденция возникновения нового мира и консолидации БРИКС необратима, и Венесуэла скоро войдет в БРИКС», — подчеркнул он.

По словам лидера республики, новый мир с БРИКС уже существует и необратим. Мадуро отметил, что он пришел на смену «старому колониальному миру с войнами, интервенциями, геноцидом и комплексом превосходства».

Ранее президент Венесуэлы заявил, что Россия побеждает Запад в развязанном им же конфликте на Украине. Он уточнил, что «эта война оказалась невыносима для США».

Губернатор Белгородской области Вячеслав Гладков предупредил о мошенниках, которые подделывают его голос и пишут сообщения от его имени. Об этом он написал на своей странице во «ВКонтакте».

«Друзья, у меня нет такой практики с рассылкой голосовых сообщений, я этого никогда не делаю. Внимательно смотрите, с каких номеров вам пишут, приходят голосовые сообщения», — написал глава региона.

Гладков посоветовал жителям обращаться в приемную в случае, если на их телефоны приходят сообщения от имени губернатора. Помощники прояснят ситуацию и подтвердят и опровергнут те запросы, которые могут поступать, добавил он.

Ранее губернатор Красноярского края Михаил Котюков рассказал, что его пытались обмануть мошенники. Ему позвонили с незнакомого номера. Неизвестный мужчина обратился к губернатору по имени-отчеству и представился сотрудником сотового оператора. Он сообщил, что срок действия сим-карты Котюкова якобы заканчивается, после этого глава региона бросил трубку.
В конце ноября, в оттепель, часов в девять утра, поезд Петербургско-Варшавской железной дороги на всех парах подходил к Петербургу. Было так сыро и туманно, что насилу рассвело; в десяти шагах, вправо и влево от дороги, трудно было разглядеть хоть что-нибудь из окон вагона. Из пассажиров были и возвращавшиеся из-за границы; но более были наполнены отделения для третьего класса, и всё людом мелким и деловым, не из очень далека. Все, как водится, устали, у всех отяжелели за ночь глаза, все назяблись, все лица были бледно-желтые, под цвет тумана.

В одном из вагонов третьего класса, с рассвета, очутились друг против друга, у самого окна, два пассажира, – оба люди молодые, оба почти налегке, оба не щегольски одетые, оба с довольно замечательными физиономиями, и оба пожелавшие, наконец, войти друг с другом в разговор. Если б они оба знали один про другого, чем они особенно в эту минуту замечательны, то, конечно, подивились бы, что случай так странно посадил их друг против друга в третьеклассном вагоне петербургско-варшавского поезда. Один из них был небольшого роста, лет двадцати семи, курчавый и почти черноволосый, с серыми, маленькими, но огненными глазами. Нос его был широк и сплюснут, лицо скулистое; тонкие губы беспрерывно складывались в какую-то наглую, насмешливую и даже злую улыбку; но лоб его был высок и хорошо сформирован и скрашивал неблагородно развитую нижнюю часть лица. Особенно приметна была в этом лице его мертвая бледность, придававшая всей физиономии молодого человека изможденный вид, несмотря на довольно крепкое сложение, и вместе с тем что-то страстное, до страдания, не гармонировавшее с нахальною и грубою улыбкой и с резким, самодовольным его взглядом. Он был тепло одет, в широкий, мерлушечий, черный, крытый тулуп, и за ночь не зяб, тогда как сосед его принужден был вынести на своей издрогшей спине всю сладость сырой ноябрьской русской ночи, к которой, очевидно, был не приготовлен. На нем был довольно широкий и толстый плащ без рукавов и с огромным капюшоном, точь-в-точь как употребляют часто дорожные, по зимам, где-нибудь далеко за границей, в Швейцарии, или, например, в Северной Италии, не рассчитывая, конечно, при этом и на такие концы по дороге, как от Эйдткунена до Петербурга. Но что годилось и вполне удовлетворяло в Италии, то оказалось не совсем пригодным в России. Обладатель плаща с капюшоном был молодой человек, тоже лет двадцати шести или двадцати семи, роста немного повыше среднего, очень белокур, густоволос, со впалыми щеками и с легонькою, востренькою, почти совершенно белою бородкой. Глаза его были большие, голубые и пристальные; во взгляде их было что-то тихое, но тяжелое, что-то полное того странного выражения, по которому некоторые угадывают с первого взгляда в субъекте падучую болезнь. Лицо молодого человека было, впрочем, приятное, тонкое и сухое, но бесцветное, а теперь даже досиня иззябшее. В руках его болтался тощий узелок из старого, полинялого фуляра, заключавший, кажется, все его дорожное достояние. На ногах его были толстоподошвенные башмаки с штиблетами, – всё не по-русски. Черноволосый сосед в крытом тулупе все это разглядел, частию от нечего делать, и, наконец, спросил с тою неделикатною усмешкой, в которой так бесцеремонно и небрежно выражается иногда людское удовольствие при неудачах ближнего:

– Зябко?

И повел плечами.

– Очень, – ответил сосед с чрезвычайною готовностью, – и заметьте, это еще оттепель. Что ж, если бы мороз? Я даже не думал, что у нас так холодно. Отвык.

– Из-за границы, что ль?

– Да, из Швейцарии.

– Фью! Эк ведь вас!..

Черноволосый присвистнул и захохотал.

Завязался разговор. Готовность белокурого молодого человека в швейцарском плаще отвечать на все вопросы своего черномазого соседа была удивительная и без всякого подозрения совершенной небрежности, неуместности и праздности иных вопросов. Отвечая, он объявил, между прочим, что действительно долго не был в России, с лишком четыре года, что отправлен был за границу по болезни, по какой-то странной нервной болезни, вроде падучей или Виттовой пляски, каких-то дрожаний и судорог. Слушая его, черномазый несколько раз усмехался; особенно засмеялся он, когда на вопрос: «что же, вылечили?» – белокурый отвечал, что «нет, не вылечили».

– Хе! Денег что, должно быть, даром переплатили, а мы-то им здесь верим, – язвительно заметил черномазый.

– Истинная правда! – ввязался в разговор один сидевший рядом и дурно одетый господин, нечто вроде закорузлого в подьячестве чиновника, лет сорока, сильного сложения, с красным носом и угреватым лицом, – истинная правда-с, только все русские силы даром к себе переводят!

– О, как вы в моем случае ошибаетесь, – подхватил швейцарский пациент, тихим и примиряющим голосом, – конечно, я спорить не могу, потому что всего не знаю, но мой доктор мне из своих последних еще на дорогу сюда дал, да два почти года там на свой счет содержал.

– Что ж, некому платить, что ли, было? – спросил черномазый.

– Да, господин Павлищев, который меня там содержал, два года назад помер; я писал потом сюда генеральше Епанчиной, моей дальней родственнице, но ответа не получил. Так с тем и приехал.

– Куда же приехали-то?

– То есть где остановлюсь?.. Да не знаю еще, право… так…

– Не решились еще?

И оба слушателя снова захохотали.

– И небось в этом узелке вся ваша суть заключается? – спросил черномазый.

– Об заклад готов биться, что так, – подхватил с чрезвычайно довольным видом красноносый чиновник, – и что дальнейшей поклажи в багажных вагонах не имеется, хотя бедность и не порок, чего опять-таки нельзя не заметить.

Оказалось, что и это было так: белокурый молодой человек тотчас же и с необыкновенною поспешностью в этом признался.

– Узелок ваш все-таки имеет некоторое значение, – продолжал чиновник, когда нахохотались досыта (замечательно, что и сам обладатель узелка начал наконец смеяться, глядя на них, что увеличило их веселость), – и хотя можно побиться, что в нем не заключается золотых, заграничных свертков с наполеондорами и фридрихсдорами, ниже с голландскими арапчиками, о чем можно еще заключить, хотя бы только по штиблетам, облекающим иностранные башмаки ваши, но… если к вашему узелку прибавить в придачу такую будто бы родственницу, как, примерно, генеральша Епанчина, то и узелок примет некоторое иное значение, разумеется, в том только случае, если генеральша Епанчина вам действительно родственница, и вы не ошибаетесь, по рассеянности… что очень и очень свойственно человеку, ну хоть… от излишка воображения.

'''
translate_and_evaluate_rus_to_eng(plain_text_rus)

#BLEU score: 0.2820121312002901


# pip install datasets transformers nltk sacrebleu
from transformers import pipeline
from datasets import load_dataset
import sacrebleu
import torch
import nltk

nltk.download('punkt')

def test_on_flores_dataset(src_lang_code: str, tgt_lang_code: str, src_lang: str, tgt_lang: str):
    # Load the FLORES dataset
    flores_dataset = load_dataset("facebook/flores", src_lang_code)["devtest"]
    
    # Determine the model based on the source and target languages
    if src_lang == "English" and tgt_lang == "Russian":
        model_name = "Helsinki-NLP/opus-mt-en-ru"
    elif src_lang == "Russian" and tgt_lang == "English":
        model_name = "Helsinki-NLP/opus-mt-ru-en"
    else:
        raise ValueError("Unsupported language pair")

    # Load the translation pipeline
    translation_pipeline = pipeline("translation", model=model_name, device=0 if torch.cuda.is_available() else -1)

    # Extract sentences from the dataset
    sentences = [example['sentence'] for example in flores_dataset]

    # Translate sentences
    translated_texts = [translation_pipeline(sentence)[0]['translation_text'] for sentence in sentences]

    # Calculate the BLEU score
    references = [[example['sentence']] for example in flores_dataset]
    bleu_score = sacrebleu.corpus_bleu(translated_texts, references).score
    print(f"BLEU score for {src_lang} to {tgt_lang}: {bleu_score}")

# Test the function with FLORES dataset for English to Russian translation
test_on_flores_dataset("eng_Latn", "rus_Cyrl", "English", "Russian")
#BLEU score for English to Russian: 5.809665204409192

# Test the function with FLORES dataset for Russian to English translation
test_on_flores_dataset("rus_Cyrl", "eng_Latn", "Russian", "English")

#BLEU score for Russian to English: 5.816635421147515


from google.cloud import translate_v2 as google_translate
import sacrebleu
from datasets import load_dataset
from yandexfreetranslate import YandexFreeTranslate

# # Initialize Google Translate API client
# google_client = google_translate.Client()

# # Initialize Yandex Free Translate
# yandex_client = YandexFreeTranslate(api="ios")

eng_eng = '''

The two-day conference, which began on Sunday, was the second of its kind. It was held to try to chart a course forward for international engagement with the country. But the Taliban administration took issue with the inclusion of some groups at the meeting. Attended by special envoys from 25 countries and regional organizations, the conference is aimed at increasing international engagement with Afghanistan and developing a more coordinated response to the problems afflicting the war-torn nation.

The Taliban administration, the de facto rulers of Afghanistan since 2021, had been invited to the conference but at the last minute the group said it would not attend. In a statement, the Taliban’s Ministry of Foreign Affairs said it should be the sole official representative of Afghanistan for talks with the international community and only then could engage in frank discussions. Inclusion of others would hinder progress, the statement added.

“This government of Afghanistan cannot be coerced by anyone,” it stated.

Representatives from Afghan civil society, women’s groups, the Organization of Islamic Cooperation, the European Union and the Shanghai Cooperation Organization were present at the conference. Afghan political opposition parties, including the National Resistance Front, which has a small armed wing, were not invited, although they had asked to be included.

The Taliban administration’s decision, announced on the eve of the conference, appeared to have been made to avoid awkward conversations with Afghans living outside the country who oppose the Taliban authorities’ exclusion of women, and political opponents inside Afghanistan, several delegates said.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.

Regardless of who said what and when, Carragher — who played for Liverpool until he retired at age 35 — is not a lone voice in this debate, and Fabinho and Casemiro are far from the only players singled out for seemingly having lead in their boots.

Any footballer over the age of 30 who is struggling for form leaves themselves open to that type of criticism, but in particular if they are now coming off second best in the sort of duels they used to win and playing in a way that makes it look like the game is now a split-second too quick for them.
Everyone agreed that it was worth more than all the wealth of the kingdom: but the king said, 'One feather is of no use to me, I must have the whole bird.' Then the gardener's eldest son set out and thought to find the golden bird very easily; and when he had gone but a little way, he came to a wood, and by the side of the wood he saw a fox sitting; so he took his bow and made ready to shoot at it. Then the fox said, 'Do not shoot me, for I will give you good counsel; I know what your business is, and that you want to find the golden bird. You will reach a village in the evening; and when you get there, you will see two inns opposite to each other, one of which is very pleasant and beautiful to look at: go not in there, but rest for the night in the other, though it may appear to you to be very poor and mean.' But the son thought to himself, 'What can such a beast as this know about the matter?' So he shot his arrow at the fox; but he missed it, and it set up its tail above its back and ran into the wood. Then he went his way, and in the evening came to the village where the two inns were; and in one of these were people singing, and dancing, and feasting; but the other looked very dirty, and poor. 'I should be very silly,' said he, 'if I went to that shabby house, and left this charming place'; so he went into the smart house, and ate and drank at his ease, and forgot the bird, and his country too. Time passed on; and as the eldest son did not come back, and no tidings were heard of him, the second son set out, and the same thing happened to him. He met the fox, who gave him the good advice: but when he came to the two inns, his eldest brother was standing at the window where the merrymaking was, and called to him to come in; and he could not withstand the temptation, but went in, and forgot the golden bird and his country in the same manner. Time passed on again, and the youngest son too wished to set out into the wide world to seek for the golden bird; but his father would not listen to it for a long while, for he was very fond of his son, and was afraid that some ill luck might happen to him also, and prevent his coming back.
'''

translated_eng_google = '''
Двухдневная конференция, начавшаяся в воскресенье, стала второй в своем роде. Оно было проведено с целью попытаться наметить дальнейший курс международного взаимодействия со страной. Однако администрация Талибана не согласилась с включением некоторых групп на встречу. Конференция, в которой принимают участие специальные посланники из 25 стран и региональных организаций, направлена на расширение международного взаимодействия с Афганистаном и разработку более скоординированного ответа на проблемы, от которых страдает раздираемая войной страна.

Администрация Талибана, де-факто правящая Афганистаном с 2021 года, была приглашена на конференцию, но в последнюю минуту группа заявила, что не будет присутствовать. В заявлении министерства иностранных дел Талибана говорится, что оно должно быть единственным официальным представителем Афганистана на переговорах с международным сообществом и только тогда сможет участвовать в откровенных дискуссиях. Включение других будет препятствовать прогрессу, добавлено в заявлении.

«Никто не может принуждать это правительство Афганистана», — заявило оно.

На конференции присутствовали представители афганского гражданского общества, женских групп, Организации исламского сотрудничества, Европейского Союза и Шанхайской организации сотрудничества. Афганские политические оппозиционные партии, в том числе Фронт национального сопротивления, имеющий небольшое вооруженное крыло, не были приглашены, хотя они просили об их включении.

Решение администрации Талибана, объявленное накануне конференции, похоже, было принято для того, чтобы избежать неловких разговоров с афганцами, живущими за пределами страны, которые выступают против исключения женщин властями Талибана, и политическими оппонентами внутри Афганистана, заявили несколько делегатов.

Независимо от того, кто, что и когда сказал, Каррагер, который играл за «Ливерпуль» до тех пор, пока не завершил карьеру в возрасте 35 лет, не является единственным голосом в этих дебатах, а Фабиньо и Каземиро — далеко не единственные игроки, выделенные за то, что, по-видимому, лидируют в своих ботинках.

Любой футболист старше 30 лет, который борется за форму, оставляет себя открытым для такого рода критики, но особенно если он сейчас занимает второе место в тех поединках, в которых он раньше выигрывал, и играет так, чтобы это выглядело как будто игра теперь для них на долю секунды слишком быстрая.

Независимо от того, кто, что и когда сказал, Каррагер, который играл за «Ливерпуль» до тех пор, пока не завершил карьеру в возрасте 35 лет, не является единственным голосом в этих дебатах, а Фабиньо и Каземиро — далеко не единственные игроки, выделенные за то, что, по-видимому, лидируют в своих ботинках.

Любой футболист старше 30 лет, который борется за форму, оставляет себя открытым для такого рода критики, но особенно если он сейчас занимает второе место в тех поединках, в которых он раньше выигрывал, и играет так, чтобы это выглядело как будто игра теперь для них на долю секунды слишком быстрая.
Все согласились, что оно стоит больше, чем все богатства королевства, но король сказал: «Одно перо мне бесполезно, мне нужна целая птица». Тогда старший сын садовника отправился в путь и думал, что очень легко найдет золотую птицу; и когда он прошёл совсем немного, он подошел к лесу и увидел возле леса сидящую лису; поэтому он взял свой лук и приготовился стрелять в него. Тогда лиса сказала: «Не стреляйте в меня, я дам вам хороший совет; Я знаю, чем ты занимаешься, и что ты хочешь найти золотую птицу. Вечером вы доберетесь до деревни; и когда доберешься туда, то увидишь два постоялых двора напротив друг друга, один из которых очень приятен и красив на вид: не ходи туда, а отдохни на ночь в другом, хотя он может показаться тебе очень бедный и подлый». Но сын подумал про себя: «Что такое животное может знать об этом?» Итак, он выпустил стрелу в лису; но он промахнулся, и он поднял хвост над спиной и побежал в лес. Затем он пошел своей дорогой и вечером пришел в деревню, где были две гостиницы; и в одном из них люди пели, танцевали и пировали; но другой выглядел очень грязным и бедным. «Я был бы очень глуп, — сказал он, — если бы пошел в этот ветхий дом и покинул это очаровательное место»; поэтому он вошел в умный дом, ел и пил, не торопясь, и забыл птицу и свою страну тоже. Время шло; а так как старший сын не вернулся и не было о нем известий, то второй сын отправился в путь, и с ним случилось то же самое. Он встретил лису, которая дала ему добрый совет; но когда он пришел к двум гостиницам, его старший брат стоял у окна, где шло веселье, и звал его, чтобы он вошел; и он не выдержал искушения, но вошел и точно так же забыл золотую птицу и свою страну. Время снова шло, и младший сын тоже захотел отправиться в большой мир искать золотую птицу; но отец его долго не слушал этого, так как очень любил сына и боялся, как бы и с ним не случилось несчастье и не помешало бы ему вернуться.
'''

translated_eng_yandex = '''
Двухдневная конференция, которая началась в воскресенье, была второй в своем роде. Она была проведена с целью попытаться наметить дальнейший курс международного взаимодействия со страной. Но администрация движения "Талибан" выступила против участия некоторых групп во встрече. Конференция, в которой принимают участие специальные посланники из 25 стран и региональных организаций, направлена на расширение международного взаимодействия с Афганистаном и выработку более скоординированных мер реагирования на проблемы, с которыми сталкивается истерзанная войной нация.

Администрация движения "Талибан", де-факто правящего Афганистаном с 2021 года, была приглашена на конференцию, но в последнюю минуту группа заявила, что не будет присутствовать. В заявлении Министерства иностранных дел движения "Талибан" говорится, что оно должно быть единственным официальным представителем Афганистана на переговорах с международным сообществом и только после этого может участвовать в откровенных дискуссиях. Привлечение других помешало бы прогрессу, добавляется в заявлении.

“Это правительство Афганистана никем не может быть принуждено”, - говорилось в нем.

На конференции присутствовали представители гражданского общества Афганистана, женских групп, Организации исламского сотрудничества, Европейского союза и Шанхайской организации сотрудничества. Афганские политические оппозиционные партии, включая Фронт национального сопротивления, у которого есть небольшое вооруженное крыло, не были приглашены, хотя они просили о включении.

Решение администрации движения "Талибан", объявленное накануне конференции, по-видимому, было принято для того, чтобы избежать неловких разговоров с афганцами, живущими за пределами страны, которые выступают против исключения женщин властями движения "Талибан", и политическими оппонентами внутри Афганистана, заявили несколько делегатов.

Независимо от того, кто что сказал и когда, Каррагер, который играл за "Ливерпуль" до тех пор, пока не ушел на пенсию в возрасте 35 лет, — не одинокий голос в этой дискуссии, и Фабиньо и Каземиро далеко не единственные игроки, выделенные за то, что, казалось бы, у них есть преимущество.

Любой футболист старше 30 лет, который борется за форму, оставляет себя открытым для такого рода критики, но, в частности, если он сейчас занимает второе место в тех поединках, которые раньше выигрывал, и играет так, что кажется, что игра теперь тоже длится доли секунды быстро для них.

Независимо от того, кто что сказал и когда, Каррагер, который играл за "Ливерпуль" до тех пор, пока не ушел на пенсию в возрасте 35 лет, — не одинокий голос в этой дискуссии, и Фабиньо и Каземиро далеко не единственные игроки, выделенные за то, что, казалось бы, у них есть преимущество.

Любой футболист старше 30 лет, который борется за форму, оставляет себя открытым для такого рода критики, но, в частности, если он сейчас занимает второе место в тех поединках, которые раньше выигрывал, и играет так, что кажется, что игра теперь тоже длится доли секунды быстро для них.
Все согласились, что оно стоит больше, чем все богатства королевства, но король сказал: "Одно перо мне ни к чему, я должен получить птицу целиком". Тогда старший сын садовника отправился в путь и подумал, что найти золотую птицу очень легко; и когда у него было пройдя совсем немного, он пришел в лес и увидел на опушке сидящую лису; тогда он взял свой лук и приготовился выстрелить в нее. Тогда лиса сказала: "Не стреляй в меня, потому что я дам тебе хороший совет; я знаю, в чем твое дело, и что ты хочешь найти золотую птицу. Вечером ты доберешься до деревни.; и когда ты доберешься туда, ты увидишь две гостиницы напротив друг друга, на одну из которых очень приятно и красиво смотреть: не заходи туда, а отдохни на ночь в другой, хотя она может показаться тебе очень бедной и подлой". Но сын подумал про себя: "Что может знать об этом деле такое чудовище, как это?" И он пустил свою стрелу в лису, но промахнулся, и она задрала хвост выше спины и убежала в лес. Затем он пошел своей дорогой и вечером пришел в деревню, где были две гостиницы; и в одной из них люди пели, танцевали и пировали; но другой выглядел очень грязным и бедным. "Я был бы очень глуп, - сказал он, - если бы пошел в этот убогий дом и покинул это очаровательное место"; и он пошел в шикарный дом, и ел, и пил в свое удовольствие, и забыл о птице, и о своей стране тоже. Время шло; и так как старший сын не возвращался, и о нем не было слышно никаких вестей, второй сын отправился в путь, и с ним случилось то же самое. Он встретил лису, которая дала ему хороший совет: но когда он подошел к двум постоялым дворам, его старший брат стоял у окна, где было веселье, и позвал его войти; и он не смог устоять перед искушением, но вошел внутрь и точно так же забыл золотую птицу и свою страну. Снова прошло время, и младший сын тоже захотел отправиться в большой мир на поиски золотой птицы; но его отец долго не хотел этого слушать, потому что он очень любил своего сына и боялся, что с ним тоже может случиться какое-нибудь несчастье, и предотвратить его возвращение.
'''

import nltk
import sacrebleu

nltk.download('punkt')

# Function to divide text into sentences
def divide_into_sentences(text):
    return nltk.tokenize.sent_tokenize(text)

# Original English text divided into sentences
original_sentences = divide_into_sentences(eng_eng)

# Translated sentences divided into sentences
translated_sentences_google = divide_into_sentences(translated_eng_google)
translated_sentences_yandex = divide_into_sentences(translated_eng_yandex)

# Evaluate translations using sacrebleu
def evaluate_translations(original, translations):
    # Ensure the number of sentences match for a fair comparison
    min_length = min(len(original), len(translations))
    original = original[:min_length]
    translations = translations[:min_length]
    
    # Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(translations, [original]).score
    return bleu_score

# Calculate BLEU scores
bleu_score_google = evaluate_translations(original_sentences, translated_sentences_google)
bleu_score_yandex = evaluate_translations(original_sentences, translated_sentences_yandex)

print(f"BLEU score using Google Translate: {bleu_score_google}")
print(f"BLEU score using Yandex Free Translate: {bleu_score_yandex}")
# BLEU score using Google Translate: 0.1209268439353369
# BLEU score using Yandex Free Translate: 0.11069375909194523