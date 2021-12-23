# Rating toxic comments

**Disclaimer** : this project contains **vulgar** text.

In this project we are going to classify some comments of the Wikipedia Talk page. It is the fourth competition of jigsaw of classifying comments. They ask workers to judge individually 2 comments. They need to say which one is the less toxic and which one is the most. Asking whereas a comment is toxic or not is difficult as each person as his own sense of toxicity.


We need to score 14 000 comments based on their toxicity. The particularity of this data, is that we do not have a properly training dataset where we have the comments and their toxicity score. We then need to find a way to train our dataset.

## Some descriptive statistics


## How to build our training set ?
Our first idea was to use the workers judgement. We concatenated the less toxic comments and the most toxic comments in the same column. We gave a score of 1 if the worker said it was toxic and 0 if he said it was less toxic.
But the basic idea is that one worker could have said that one comment is less toxic and that another said that is was the most toxic. So we aggregated the dataset by comments and took the mean of the score. 
For instance a comment that has a score of 0.6 means that 60% of the workers who needed to classify this comment said that it was toxic.
So we ended with a training set of around 14 000 comments.
