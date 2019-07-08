import pandas as pd 
import numpy as np
import os
import pickle
import csv
from sklearn.preprocessing import MultiLabelBinarizer


def train():
    print("Training model...")

    # Load data and set id as our index
    df_food_data = pd.read_json(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", "train.json"))
    df_food_data.set_index('id', inplace=True)
    df_food_data = df_food_data.head(1000)

    # One Hot Encoding
    df_ingredient = df_food_data['ingredients']
    mlb = MultiLabelBinarizer()
    df_ingredient = pd.DataFrame(mlb.fit_transform(df_ingredient),columns=mlb.classes_, index=df_food_data.index)
    
    # TODO: we need to find the normalized term frequency of each recipe-ingredient
    df_ingredient_normalized = df_ingredient

    # Now we get the user ratings for each recipe
    df_user_ratings = pd.read_csv(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                "content_based_meal_recommendations", "data", "user_preferences.csv"))

    rating = pd.pivot_table(df_user_ratings, values='rating', index=['food_id'], columns = ['user_id']).T
    rating.fillna(0, inplace=True)

    # We find the user's preferences for each ingredient (create user_profile)
    df_ingredient_filtered = df_ingredient_normalized[df_ingredient_normalized.index.isin(rating.columns.values)].sort_index()

    user_recipe_matrix = rating.values.dot(df_ingredient_filtered.values)
    user_preferences = pd.DataFrame(user_recipe_matrix, columns=df_ingredient_filtered.columns, index=rating.index)
    
    # Calculate IDF
    document_frequency = df_ingredient.sum()
    idf = (len(df_food_data)/document_frequency).apply(np.log)
    idf_df_item = df_ingredient.mul(idf.values)

    # Find the recipes that most match the user's preferences
    users_no = rating.index
    df_predict = pd.DataFrame()
    for i in (range(len(users_no))):
        working_df = idf_df_item.mul(user_preferences.iloc[i], axis=1)
        df_predict[users_no[i]] = working_df.sum(axis=1)

    with open(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", 'tfidf_model.pkl'), 'wb') as f:
        pickle.dump(df_predict, f)
    
    print('Finished training model!\n')
    return df_predict.copy()


def predict(df_predict, user_no):
    print('Predicting for user id, ' + str(user_no))
    # Load data
    df_food_data = pd.read_json(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", "train.json"))
    df_food_data.set_index('id', inplace=True)
    df_food_data = df_food_data.head(1000)

    #user predicted rating to all books
    if (user_no not in df_predict.columns.values):
        return cold_start(df_food_data.sample(n=3), user_no)
    user_predicted_rating = df_predict[user_no]

    #combine book rating and book detail
    user_rating_food = pd.concat([user_predicted_rating,df_food_data], axis=1)

    print('Done predicting!\n')
    return user_rating_food.sort_values(by=[user_no], ascending=False).iloc[0:10]


# TODO: cold_start should update online such that we don't have to retrain the entire model
def cold_start(df_food_data, user_no):
    """ In content-based models, the cold start problem are usually solved in one of the following ways:
            1. The user is recommended the most popular items 
            2. The user is recommended things based on other metadata (program, location)
            3. The user is given several recipes and asked to rate them
        For this implementation, we will choose option 3.
    """
    preferences = list()
    print("\nWe don't have any information about you. Please answer a few questions.")
    for index, row in df_food_data.iterrows():
        print(row)
        rating = input("\nWhat do you think of the recipe above? 1 = like, 0 = neutral, -1 = dislike\n")
        preference = (user_no, index, int(rating))
        preferences.append(preference)

    # Update the user preference csv
    # TODO: Transition to online model so this isn't necessary
    with open(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", 'user_preferences.csv'),'a') as f:
        writer = csv.writer(f)
        writer.writerows(preferences)

    print("Thanks! The model will be retrained with your preferences")
    df_predict = train()
    return predict(df_predict, user_no)

if __name__ == "__main__":

    #update_preferences(1, 70, -1)

    # train the model based on preferences
    train()

    # load the model
    df_predict = pd.read_pickle(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", 'tfidf_model.pkl'))
    # print(df_predict)
    while True:
        user_id = input("\nPlease enter your user id\n")
        if user_id.isnumeric():
            print(predict(df_predict, int(user_id)))
        else:
            break