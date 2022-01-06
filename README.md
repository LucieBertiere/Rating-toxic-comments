# Rating toxic comments

**Disclaimer** : this project contains **vulgar** text.

In this project we are going to classify some comments of the Wikipedia Talk page. It is the fourth competition of jigsaw to classify comments. They ask workers to judge individually 2 comments. They need to say which one is the less toxic and which one is the most. Asking whereas a comment is toxic or not is difficult as each person as his own sense of toxicity.


We need to score 14 000 comments based on their toxicity. The particularity of this data, is that we do not have a properly training dataset where we have the comments and their toxicity score. We then need to find a way to train our dataset.

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
Here is the wordcloud of the comments we need to score.

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/comments_to_score.PNG" width="450" height="275">
  
</div>

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
We used ridge regression as it is well known to work when we have a lot of predictors (specifically when we have more predictors than observations) and that these predictors are colinear. It is able to tell the difference between useful and unuseful predictors, which helps to avoid overfitting and increase the accuracy.
Using this type of regression helped us to improve our score, increasing by 0.08 with only the default parameters of the ridge regression. Then we decided to control some parameter such as the precision or the regularization. This helped us to increase the score by 0.008 related to the basic regression. 

<div align="center">
  
|      **Model**      | **Accuracy** |
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

#### Using SVD :

#### Using Ridge Regression :
Using the same Ridge model as the one we did with the first idea, we increase our score by almost 0.01.

But then we thought about giving more weights to really toxic comments (i.e. those who are labelled as severe toxic, threatning or having identity hate). This would have the power to make our model distinguish better between non-toxic and toxic comments. This increased our model accuracy by 0.016.

Finally using this weighted ridge regression, we decided to ensemble 3 different of this model, controlling the regularization parameter $$\alpha$$, to make our accuracy better. 
We used :
- $$\alpha = 10$$
- $$\alpha = 1$$
- $$\alpha = 0.5$$



## Bibliography
(1) https://colah.github.io/posts/2015-08-Understanding-LSTMs/

(2) https://corporatefinanceinstitute.com/resources/knowledge/other/ridge/
