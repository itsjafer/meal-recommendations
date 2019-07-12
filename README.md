# Content-based model proof of concept

This recommender will take user preferences for *recipes* and use these preferences to find the user's preferences for *ingredients*. Using the ingredients, we are then able to search among all recipes which have a similar ingredient matrix to that which the user prefers.

We will use TF-IDF as the measure of similarity between recipes. 83% of text-based recommenders use TF-IDF.

## Getting Started

Install python 3.6.3:

```pyenv install 3.6.3```

Create `content_poc` virtual env:

```pyenv virtualenv 3.6.3 content_poc```

Install requirements:

```pip install -r requirements.txt```

Run the main file. You will be asked for a user id; feel free to give it any number you like (as long as you can remember it):

```python content_based_meal_recommendation_main.py```

You will be given 3 recipes to rate as a cold-start. Accepted inputs are `1,0,-1`.

You will then be given a list of recipes that match your preferences best. If you enter your user id again, you can choose to rate more recipes for more accurate results or just retrieve predictions.

## Requirements

* Python 3.6.3
* Pandas
* scikit-learn

## Dataset

`train.json` is a dataset consisting of a list of recipes where each recipe has a list of ingredients as well as an associated cuisine. We do not use the cuisine as a factor in training although it's useful in proving that our predictions are accurate.

## TF-IDF

TF stands for *Term Frequency* which is the number of times a word/term appears in our dataset.

IDF stands for *Inverse Document Frequency* which is the inverse of the number of documents in which a word appears.

TF-IDF increases proportionally to the number of times a keyword appears but is offset by the number of couments in which it appears.

`TF-IDF = TF * IDF`

### Creating user profiles

We have:
1. A list of recipes
2. User ratings for each recipe

For each ingredient in a recipe, we calculate term frequency normalized: `1/sqrt(N)` where `N` is the number of terms. We will create a user profile vector of TF-IDF values and multipy this by each recipe preference.

The main idea:

1. We convert a user's recipe preference into an ingredient preference matrix
2. We compare each recipe's ingredients to the user's ingredient preferences
3. We calculate a score for each recipe based on this

![Example](example_1.png?raw=true "A recipe-keyword matrix and a user-preference matrix")

![Example cont.](example_2.png?raw=true "Finding the user preference vector")

![Example cont.](example_3.png?raw=true "Finding the user's preference for a given recipe")
