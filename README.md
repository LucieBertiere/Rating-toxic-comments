# Rating toxic comments

**Disclaimer** : this project contains **vulgar** text.

In this project we are going to classify some comments of the Wikipedia Talk page. It is the fourth competition of jigsaw to classify comments. They ask workers to judge individually 2 comments. They need to say which one is the less toxic and which one is the most. Asking whereas a comment is toxic or not is difficult as each person as his own sense of toxicity.


We need to score 14 000 comments based on their toxicity. The particularity of this data, is that we do not have a properly training dataset where we have the comments and their toxicity score. We then need to find a way to train our dataset.

## Some descriptive statistics
Here is the wordcloud of the comments we need to score.

<div align="center">
  
<img src="https://github.com/LucieBertiere/Rating-toxic-comments/blob/main/Images/comments_to_score.PNG" width="500" height="300">
  
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
 

On the above image you can see how a chunk of neural network in LSTM is build, having 3 different gates which can control previous and input information: 

 

 

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
