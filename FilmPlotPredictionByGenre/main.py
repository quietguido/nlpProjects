import numpy as np
import pandas as pd
import os, json, gc, re, random
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# task0

import torch, transformers, tokenizers
torch.__version__, transformers.__version__, tokenizers.__version__


movies_df = pd.read_csv("../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv")
movies_df

# task1


normalize_text = [
('bio-pic', 'biography'),
('biopic', 'biography'),
('biographical', 'biography'),
('biodrama', 'biography'),
('bio-drama', 'biography'),
('biographic', 'biography'),
('animated','animation'),
('anime','animation'),
('children\'s','children'),
('comedey','comedy'),
(' set 4,000 years ago in the canadian arctic',''),
('historical','history'),
('romantic','romance'),
('3-d','animation'),
('3d','animation'),
('viacom 18 motion pictures',''),
('sci-fi','science_fiction'),
('ttriller','thriller'),
('.',''),
('based on radio serial',''),
(' on the early years of hitler',''),
('sci fi','science_fiction'),
('science fiction','science_fiction'),
(' (30min)',''),
('16 mm film','short'),
('\[140\]','drama'),
('\[144\]',''),
(' for ',''),
('adventures','adventure'),
('kung fu','martial_arts'),
('kung-fu','martial_arts'),
('martial arts','martial_arts'),
('world war ii','war'),
('world war i','war'),
('biography about montreal canadiens star|maurice richard','biography'),
('bholenath movies|cinekorn entertainment',''),
(' \(volleyball\)',''),
('spy film','spy'),
('anthology film','anthology'),
('biography fim','biography'),
('avant-garde','avant_garde'),
('biker film','biker'),
('buddy cop','buddy'),
('buddy film','buddy'),
('comedy 2-reeler','comedy'),
('films',''),
('film',''),
('biography of pioneering american photographer eadweard muybridge','biography'),
('british-german co-production',''),
('bruceploitation','martial_arts'),
('comedy-drama adaptation of the mordecai richler novel','comedy-drama'),
('movies by the mob\|knkspl',''),
('movies',''),
('movie',''),
('coming of age','coming_of_age'),
('coming-of-age','coming_of_age'),
('drama about child soldiers','drama'),
('(( based).+)',''),
('(( co-produced).+)',''),
('(( adapted).+)',''),
('(( about).+)',''),
('musical b','musical'),
('animationchildren','animation|children'),
(' period','period'),
('drama loosely','drama'),
(' \(aquatics|swimming\)',''),
(' \(aquatics|swimming\)',''),
("yogesh dattatraya gosavi's directorial debut \[9\]",''),
("war-time","war"),
("wartime","war"),
("ww1","war"),
("wwii","war"),
('unknown',''),
('psychological','psycho'),
('rom-coms','romance'),
('true crime','crime'),
('\|007',''),
('slice of life','slice_of_life'),
('computer animation','animation'),
('gun fu','martial_arts'),
('j-horror','horror'),
(' \(shogi|chess\)',''),
('afghan war drama','war drama'),
('\|6 separate stories',''),
(' \(30min\)',''),
(' (road bicycle racing)',''),
(' v-cinema',''),
('tv miniseries','tv_miniseries'),
('\|docudrama','\|documentary|drama'),
(' in animation','|animation'),
('((adaptation).+)',''),
('((adaptated).+)',''),
('((adapted).+)',''),
('(( on ).+)',''),
('american football','sports'),
('dev\|nusrat jahan','sports'),
('television miniseries','tv_miniseries'),
(' \(artistic\)',''),
(' \|direct-to-dvd',''),
('history dram','history drama'),
('martial art','martial_arts'),
('psycho thriller,','psycho thriller'),
('\|1 girl\|3 suitors',''),
(' \(road bicycle racing\)',''),
(' ','|'),
(',','|'),
('-',''),
('actionadventure','action|adventure'),
('actioncomedy','action|comedy'),
('actiondrama','action|drama'),
('actionlove','action|love'),
('actionmasala','action|masala'),
('actionchildren','action|children'),
('fantasychildren\|','fantasy|children'),
('fantasycomedy','fantasy|comedy'),
('fantasyperiod','fantasy|period'),
('cbctv_miniseries','tv_miniseries'),
('dramacomedy','drama|comedy'),
('dramacomedysocial','drama|comedy|social'),
('dramathriller','drama|thriller'),
('comedydrama','comedy|drama'),
('dramathriller','drama|thriller'),
('comedyhorror','comedy|horror'),
('sciencefiction','science_fiction'),
('adventurecomedy','adventure|comedy'),
('animationdrama','animation|drama'),
('\|\|','|'),
('muslim','religious'),
('thriler','thriller'),
('crimethriller','crime|thriller'),
('fantay','fantasy'),
('actionthriller','action|thriller'),
('comedysocial','comedy|social'),
('martialarts','martial_arts'),
('\|\(children\|poker\|karuta\)',''),
('epichistory','epic|history'),
('erotica','adult'),
('erotic','adult'),
('((\|produced\|).+)',''),
('chanbara','chambara'),
('comedythriller','comedy|thriller'),
('biblical','religious'),
('biblical','religious'),
('colour\|yellow\|productions\|eros\|international',''),
('\|directtodvd',''),
('liveaction','live|action'),
('melodrama','drama'),
('superheroes','superheroe'),
('gangsterthriller','gangster|thriller'),
('heistcomedy','comedy'),
('historic','history'),
('historydisaster','history|disaster'),
('warcomedy','war|comedy'),
('westerncomedy','western|comedy'),
('ancientcostume','costume'),
('computeranimation','animation'),
('dramatic','drama'),
('familya','family'),
('familya','family'),
('dramedy','drama|comedy'),
('dramaa','drama'),
('famil\|','family'),
('superheroe','superhero'),
('biogtaphy','biography'),
('devotionalbiography','devotional|biography'),
('docufiction','documentary|fiction'),
('familydrama','family|drama'),
('espionage','spy'),
('supeheroes','superhero'),
('romancefiction','romance|fiction'),
('horrorthriller','horror|thriller'),
('suspensethriller','suspense|thriller'),
('musicaliography','musical|biography'),
('triller','thriller'),
('\|\(fiction\)','|fiction'),
('romanceaction','romance|action'),
('romancecomedy','romance|comedy'),
('romancehorror','romance|horror'),
('romcom','romance|comedy'),
('rom\|com','romance|comedy'),
('satirical','satire'),
('horror','thriller'),
('science_fictionchildren','science_fiction|children'),
('homosexual','adult'),
('sexual','adult'),
('mockumentary','documentary'),
('periodic','period'),
('romanctic','romantic'),
('politics','political'),
('samurai','martial_arts'),
('tv_miniseries','series'),
('serial','series'),
('family','children'),
('martial_arts','action'),
('western','action'),
('western','action'),
('spy','action'),
('superhero','action'),
('suspense','action'),
('heist','action'),
('actionner','action'),
('noir','black'),
('social',''),
]


# movies_df = movies_df[(movies_df["Origin/Ethnicity"]=="American") | (movies_df["Origin/Ethnicity"]=="British")]
movies_df = movies_df[["Plot", "Genre"]]
drop_indices = movies_df[movies_df["Genre"] == "unknown" ].index
movies_df.drop(drop_indices, inplace=True)

# Combine genres: 1) "sci-fi" with "science fiction" &  2) "romantic comedy" with "romance"
movies_df["Genre"].replace(
    # {"sci-fi": "science fiction", "romantic comedy": "romance"},
    to_replace=normalize_text,
    inplace=True
    )

# Choosing movie genres based on their frequency
shortlisted_genres = movies_df["Genre"].value_counts().reset_index(name="count").query("count > 200")["index"].tolist()
movies_df = movies_df[movies_df["Genre"].isin(shortlisted_genres)].reset_index(drop=True)

# Shuffle DataFrame
movies_df = movies_df.sample(frac=1).reset_index(drop=True)

# Sample roughly equal number of movie plots from different genres (to reduce class imbalance issues)
movies_df = movies_df.groupby("Genre").head(400).reset_index(drop=True)


# Creating a LabelEncoder object for converting genre text labels into numerical values.
label_encoder = LabelEncoder()

movies_df["genre_encoded"] = label_encoder.fit_transform(movies_df["Genre"].tolist())
# Encodes genre text labels into numerical values and stores them in a new column called 'genre_encoded'.
# The mapping between genre text labels and their corresponding numerical values is stored in the LabelEncoder object.

movies_df = movies_df[["Plot", "Genre", "genre_encoded"]]
# Selecting only the 'Plot', 'Genre', and 'genre_encoded' columns for further processing.
# 'Plot' column contains the movie plot summaries.
movies_df


# task2

from simpletransformers.classification import ClassificationModel

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-cased', num_labels=len(shortlisted_genres), args=model_args)


# task3

train_df, eval_df = train_test_split(movies_df, test_size=0.2, stratify=movies_df["Genre"], random_state=42)

# Train the model
model.train_model(train_df[["Plot", "genre_encoded"]])

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df[["Plot", "genre_encoded"]])
print(result)


# task4

predicted_genres_encoded = list(map(lambda x: np.argmax(x), model_outputs))
predicted_genres = list(label_encoder.inverse_transform(predicted_genres_encoded))
eval_gt_labels = eval_df["Genre"].tolist()
class_labels = list(label_encoder.classes_)

plt.figure(figsize=(22,18))
cf_matrix = confusion_matrix(predicted_genres, eval_gt_labels, class_labels)
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, cmap="YlGnBu")
ax.set_xlabel('Predicted Genres', fontsize=20)
ax.set_ylabel('True Genres', fontsize=20)
ax.set_title('Confusion Matrix', fontsize=20)
ax.set_xticklabels(class_labels, rotation=90, fontsize=18)
ax.set_yticklabels(class_labels, rotation=0, fontsize=18)

plt.show()

from sklearn.metrics import f1_score

eval_gt_labels = eval_df["Genre"].tolist()
f1 = f1_score(eval_gt_labels, predicted_genres, average='weighted')

# Print the F1 score
print("F1 Score:", f1)


# task5

for _ in range(100):

    random_idx = random.randint(0, len(eval_df)-1)
    text = eval_df.iloc[random_idx]['Plot']
    true_genre = eval_df.iloc[random_idx]['Genre']

    # Predict with trained multiclass classification model
    predicted_genre_encoded, raw_outputs = model.predict([text])
    predicted_genre_encoded = np.array(predicted_genre_encoded)
    predicted_genre = label_encoder.inverse_transform(predicted_genre_encoded)[0]

    print(f'\nTrue Genre:'.ljust(16,' '), f'{true_genre}\n')
    print(f'Predicted Genre: {predicted_genre}\n')
    print(f'Plot: {text}\n')
    print("-------------------------------------------")



