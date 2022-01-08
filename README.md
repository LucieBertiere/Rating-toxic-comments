# Rating toxic comments

**Disclaimer** : this project contains **vulgar** text.

In this project we are going to classify some comments of the Wikipedia Talk page. It is the fourth competition of jigsaw to classify comments. They ask workers to judge individually 2 comments. They need to say which one is the less toxic and which one is the most. Asking whereas a comment is toxic or not is difficult as each person as his own sense of toxicity.
  
We need to score 14 000 comments based on their toxicity. The particularity of this data, is that we do not have a properly training dataset where we have the comments and their toxicity score. We then need to find a way to train our dataset.

This competition is evaluated using the **Average Agreement with Annotators**. The test dataset has 200 000 observations which are pair of comments said to be toxic or not by workers. It contains the 14 000 comments to score. Our prediction is then use to rank the comment pair. Let's take an example :

<div align="center">

|                    |                       **Less toxic**                      |                            **More toxic**                           |
|--------------------|:---------------------------------------------------------:|:-------------------------------------------------------------------:|
| **Pair of comment**| Untill then don't waste my time  posting crap on my page. | Shut your mouth ! Dont tell me to be  civil because I am done here. |
|**Score prediction**|                            0.9                            |                                 0.7                                 |


</div>

Here our prediction gave a score of toxicity higher to the first comment, than the second one, which is not matching what the worker judged. Thus this pair receives a 0. Inversely if the prediction was matching the worker judgement, the pair would have received a 1.

Thus it has no importance if the toxicity score is ranging from 0 to 1 or from -1 to 1 or even from -999 to 999. The idea is just to be able to compare two comments based on the predicted score.

Finally, we take the average of all this match with the predictions which gives us the ranking for kaggle.



## Cleaning the data
The first step of this project was to clean the data, to do so we created a function *cleaning_text()* :
- it puts all the text in lower case, in order to have words comparable,
- it removes URL,
- it removes #,
- it removes digits,
- it removes mentions,
- it removes punctuations,
- it removes extra white space,
- it removes stop words (i.e. the words that are the most used and that are thus not relevant to study (*I, are, have, can...*)),
- it lemmatize the text (i.e. the words that are a transformation of other words are modified  (*rocks -> rock, pulled -> pull*)).

Then we applied this function to all the comments to be able to do our models.

## Some descriptive statistics
### First dataset: Jigsaw Rate Severity of Toxic Comments
There are 7537 comments to score. Here is the wordcloud of these comments.

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/comments_to_score.PNG" width="450" height="275">
  
</div>

We can observe that the words that appear most frequently are "article", "fuck", "wikipedia", "page", "people" and "suck".

As said before, there are less and more toxic comments in the available dataset. The wordcloud corresponding to the less toxic comments is:

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/wdcloud_less.png" width="450" height="275">
  
</div>

Here, the words that appear most frequently in less toxic comments are "page", "article", "wikipedia", "please", "people" and "talk".

The wordcloud corresponding to the more toxic comments is:

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/wdcloud_more.png" width="450" height="275">
  
</div>

Here, the words that appear most frequently in more toxic comments are more agressive: "fuck", "faggot", "wikipedia", "fucking", "nigger" and "shit".

### Second dataset: Toxic Comment Classification Challenge 

Comments are ranked with respect to their content into 6 categories: **toxic**, **severe toxic**, **obscene**, **threat**, **insult** and **identity hate**. These six variables are binary, i.e., 1 if the comment is classified in that category, 0 otherwise. A comment can be classified in several categories at the same time. We have checked how many comments there are per category: 

<div align="center">

|      **Category**   |   **Total**  |
|:-------------------:|:------------:|
|       Toxic         |     15294    |
|    Severe Toxic     |      1595    |
|      Obsecne        |      8449    |
|       Threat        |       478    |
|       Insult        |      7877    |
|    Identity hate    |      1405    |


</div>

We observe that there are much more comments classified as toxic. In fact, we have noticed that all the severe toxic comments are also considered as toxic as they are both equal to 1. 

The most **frequent words** appearing in this dataset are "article", "page" and "wikipedia":

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/frequent_words.png" width="250" height="425">
  
</div>

**Unbalanced data**

We have noticed that this data is quite unbalanced by creating a score based on the sum of each of the 5 categories. By creating this new variable, we observed that almost 90% of the comments had a score of 0, which means that they were neutral because they did not belong to one of the six categories. In the following plot the different frequencies can be seen:

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/score_plot.png" width="300" height="300">
  
</div>

Hence, we had to rescale the data. 


## How to build our training set ?
### 1st idea :
Our first idea was to use the workers judgement. We concatenated the less toxic comments and the most toxic comments in the same column. We gave a score of 1 if the worker said it was toxic and 0 if he said it was less toxic.
But the basic idea is that one worker could have said that one comment is less toxic and another one can have said that it was the most toxic. So we aggregated the dataset by comments and took the mean of the score. 
For instance a comment that has a score of 0.6 means that 60% of the workers who needed to classify this comment said that it was toxic.
So we ended with a training set of around 14 000 comments.

#### Building a CNN :
As our first model we decided to build a convolutional neural network using long short term memory neural network. These neural networks permit to have the information persisting. A simple example is for understanding a sentence we need to read the words as a whole set. Reading only one word does not able us to understand the sense of the sentence. 


<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/lstm_gates.PNG" width="500" height="200">
  
</div>
 

On the above image you can see how a chunk of neural network in LSTM is build, having 3 different gates which can control previous and input information. 
To build this model we first needed to tokenize and pad the clean comments. The padded sequence was quite long, a length of over 1000 index of words. 
Our model contains 6 layers so it can learn more about the training set. The input layer is the embedding one, which helps to compress the fature into a smaller space as it transforms words into their word embeddings. Then this input goes into the LSTM layer, the natural language processing layer. Then we added three dense layers for the model to learn more about the training set. And finally we added a final dense layer which gives the prediction of the toxicity of a comment (between 0 and 1 thanks to the sigmoid activation function).

We then checked that the model had no overfitting on the test set. And we made the prediction, which was not so high a little bit more than 0.65.

#### Using Ridge Regression :
We used ridge regression as it is well known to work when we have a lot of predictors (specifically when we have more predictors than observations) and that these predictors are colinear. It is able to tell the difference between useful and unuseful predictors, putting the unuseful coefficients close to 0, which helps to avoid overfitting and increase the Average Agreement with Annotators.
Using this type of regression helped us to improve our score, increasing by 0.08 with only the default parameters of the ridge regression. Then we decided to control some parameter such as the precision or the regularization. This helped us to increase the score by 0.008 related to the basic regression. 

<div align="center">
  
|      **Model**      | **Average Agreement with Annotators** |
|:-------------------:|:------------:|
|      CNN : LSTM     |     0.657    |
| Ridge no parameters |     0.720    |
|   Ridge parameters  |     0.728    |
  
</div>

### 2nd idea :
To improve our score we decided to use another training set from a previous Jigsaw competition : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/, where each comment can be categorized (a comment can be all of this 6 categories):
- 1 if the comment is toxic
- 1 if the comment is severe_toxic
- 1 if the comment is obscene
- 1 if the comment is threat
- 1 if the comment is insult
- 1 if the comment is identity_hate

#### Using SVD and LGBM:

**SVD** is for Singular Value Decomposition which is a method based on matrix factorization and it does not assume that the matrix is quadratic (PCA does). While PCA centers the data before doing SVD, truncated SVD does not. We will use *Truncated SVD* on TF-IDF as it works on TF-IDF matrices as returned by the vectorizers on text classification. We do that to implement later the LGBM model.

**Light GBM** is a case of gradient boosting using a tree-based learning algorithm and can be used for classification and regression problems. This algorithm has many advantages such as a faster training speed, higher efficiency, higher accuracy, less expensive in terms of calculation, handle large datasets, etc. The algorithms based on trees grow the tree horizontally while LGBM grows the tree vertically. The latter grows the tree by leaves. 
There are some important parameters to be tuned: *max_depth* which corresponds to the maximum tree depth (*default* = -1); *learning_rate* which corresponds to the boosting learning rate and will determine the impact of each tree on the final output (*default* = 0.1); *num_leaves* corresponding to the maximum tree leaves in the tree (*default* = 31); *colsample_bytree* corresponding to the subsample ratio of columns when each tree is constructed (*default* = 1); *subsample* corresponding to the sub-sample report of the training body (*default* = 1); *reg_alpha* which is the L1 regularization term on weights (*default* = 0); *reg_lambda* which is the same as for *reg_alpha* but with the L2 regularization term on weights (*default* = 0); *n_estimators* which is the number of boosted trees to fit (*default* = 100). 
By tuning this parameters we obtain an Average Agreement with Annotators equal to 0.758


#### Using Ridge Regression :
Using the same Ridge model as the one we did with the first idea, we increase our score by almost 0.01 having a score of 0.767.

But then we thought about giving more weights to really toxic comments (i.e. those who are labelled as severe toxic, threatning or having identity hate). This would have the power to make our model distinguish better between non-toxic and toxic comments. This increased our model Average Agreement with Annotators by 0.016 (a total score of 0.783).

Finally using this weighted ridge regression, we decided to ensemble 3 different models, controlling the regularization parameter α, to make our Average Agreement with Annotators better. 

We used :
- α = 10 : the significance of the predictors must be high,
- α = 1
- α = 0.5 : the significance of the predictors can be low.

It is important to not use a too big α as it would tend to make all of our coefficients close to 0.

Using this ensemble method helped us to go over the 0.8 average agreement with the workers, and increase it by 0.02.

#### Using Linear Regression

Linear Regression is a supervized machine learning method where the model is going to find between the independent and dependent variables the best fit linear line. The advantage of this algorithm is that it is very easy to implement. It models a target value, here the score explained in the following lines. 
For this model, we created a score being the sum of the six categories of comments and we added more weights (weights = 3) to the comments of *severe toxic*, *threat* and *identity hate* to give them more importance in the data. We only kept 25 000 words to train our model at first, and then we tried with the first 30 000 words. However, by increasing the number of words, the Average Agreement with Annotators deteriorates: from 0.685 to 0.676. We also tried with all the words in the dataset and the Average Agreement with Annotators improved with respect to the two other cases: we got 0.686.



We have the same results as in most of the papers, the Ridge Regression tends to have a better Average Agreement with Annotators thanks to its coefficients selection.


<div align="center">
  
|          **Model**          | **Average Agreement with Annotators** |
|:---------------------------:|:------------:|
|           SVD-LGBM          |     0.758    |
|            Ridge            |     0.767    |
|      Ridge with weights     |     0.783    |
| Ensemble Ridge with weights |     0.803    |
|       LR with weights       |     0.686    |
  
</div>

## Conclusion 

Rating toxicity of comments can be very useful for social media, or web pages to limit insults, threats and so on. These models can be a first barrier to the publishment of toxic comments by censoring them.

We built different models, which were all having an Average Agreement with Annotators greater than 65%. But overall our best model was the ensemble of 3 different Ridge regression. It was one of the best model as we have a lot of words and it permits to put some coefficients close to 0 if the word is not useful to our model.

<div align="center">
  
| **Ideas** |          **Model**          | **Average Agreement with Annotators** |
:----------:|:---------------------------:|:------------:|
|   Idea 1  |          CNN : LSTM         |     0.657    |
|   Idea 1  |      Ridge no parameters    |     0.720    |
|   Idea 1  |        Ridge parameters     |     0.728    |
|   Idea 2  |           SVD-LGBM          |     0.758    |
|   Idea 2  |            Ridge            |     0.767    |
|   Idea 2  |      Ridge with weights     |     0.783    |
| **Idea 2**|**Ensemble Ridge with weights**|   **0.803**  |
|   Idea 2  |       LR with weights       |     0.686    |
  
</div>

## Bibliography
(1) https://colah.github.io/posts/2015-08-Understanding-LSTMs/

(2) https://corporatefinanceinstitute.com/resources/knowledge/other/ridge/

(3) https://medium.com/@invest_gs/classifying-tweets-with-lightgbm-and-the-universal-sentence-encoder-2a0208de0424

(4) https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a

(5) https://github.com/microsoft/LightGBM/issues/3511

(6) https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/

(7) http://pavelbazin.com/post/linear-regression-hyperparameters/#hyperparameters


