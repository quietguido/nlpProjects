Presentation: [11_12.pdf](https://github.com/quietguido/nlpProjects/files/15390291/11_12.pdf)

GENRE PREDICTION FROM WIKIPEDIA MOVIE PLOTS

TASK DESCRIPTION:

This research explores the possibility of using the BERT transformer model to predict movie genres based on their plot summaries extracted from Wikipedia.

Data Preprocessing: 

Cleaning and normalizing the movie plot data from the Wikipedia movie plots dataset. This includes filtering out irrelevant genres. 

Model Selection and Training: 

Utilizing the pre-trained BERT model (bert-base-cased) for genre classification. The model is fine-tuned on the prepared movie plot and genre data.

Evaluation:

Assessing the model's performance through metrics like F1 score and confusion matrix visualization.

DATASET:

Plot summary descriptions scraped from Wikipedia
Number of Rows in the dataset: 34886
<img width="761" alt="Screenshot 2024-05-21 at 18 13 48" src="https://github.com/quietguido/nlpProjects/assets/158144248/d001cc95-11a3-4e53-acff-bac99431aca1">

METHODS: DATA PREPROCESSING

Cleaning the Plot Summaries for Genre Prediction. 
The initial step of our research involved cleaning and normalizing the movie plot data obtained from the Wikipedia movie plots dataset. This crucial moment ensured the quality and proper format of the data for training our genre prediction model. 

Key data pre-processing step were:

<img width="1193" alt="Screenshot 2024-05-21 at 18 15 14" src="https://github.com/quietguido/nlpProjects/assets/158144248/417c483f-155d-4abe-93f0-5e88aafbf147">

METHODS: BALANCING GENRE REPRESENTATION

To prevent the model from being biased towards genres with a high number of entries, we focused on genres with a minimum number of instances. Test showed that in terms of training time and accuracy genres with minimum 200 films was proper choice for training, but we only considered maximum of 500 films from each genre.

Number of Rows in the dataset: 34886

Number of Uniqe genres in dataset before normalization: 2265

after number dropped to around 1119.

Average plot lenght: 490.56

Filter of origin as British and American: Main reasoning of this choice was to limit choices to the most popular ones.

MODEL SELECTION AND TRAINING:

Utilizing the pre-trained BERT model (bert-base-cased) for genre classification. The model is fine-tuned on the prepared movie plot and genre data. Training parameters are set to optimize performance within a limited time frame.

RESULTS:

F1 Score: 0.538538.

<img width="924" alt="Screenshot 2024-05-21 at 18 17 53" src="https://github.com/quietguido/nlpProjects/assets/158144248/96a1cd7e-3e7e-4be4-b5ef-7003f4372684">

Test Case:

True Genre: romance

Predicted Genre: drama

Plot: Mia Hall and her family are getting ready to go on with their normal day activities when it is announced on the radio that school has been canceled. Mia's dad Denny is a teacher and as a result of the snow day does not have to go to work. It is also revealed that Mia is dating an older, popular up-and-coming rockstar named Adam Wilde.


