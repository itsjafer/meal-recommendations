import pandas as pd 
import numpy as np
import os
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

def train():
    # Load data and set id as our index
    df_food_data = pd.read_json(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", "train.json"))
    df_food_data.set_index('id', inplace=True)
    df_food_data = df_food_data.head(1000)

    # One Hot Encoding
    df_ingredient = df_food_data['ingredients']
    mlb = MultiLabelBinarizer()
    df_ingredient = pd.DataFrame(mlb.fit_transform(df_ingredient),columns=mlb.classes_, index=df_food_data.index)
    print(df_ingredient)
    # Now we need to find the normalized term frequency of each recipe-ingredient

    df_ingredient_normalized = df_ingredient

    # Now we get the user ratings for each recipe
    df_user_ratings = pd.read_csv(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                "content_based_meal_recommendations", "data", "user_preferences.csv"))

    rating = pd.pivot_table(df_user_ratings, values='rating', index=['food_id'], columns = ['user_id']).T
    rating.fillna(0, inplace=True)
    rating = rating.iloc[:, ::-1]
    print(rating)

    # We find the user's preferences for each ingredient (create user_profile)
    df_ingredient_filtered = df_ingredient_normalized[df_ingredient_normalized.index.isin(rating.columns.values)]

    print(df_ingredient_filtered.loc[:, (df_ingredient_filtered != 0).any(axis=0)])
    user_recipe_matrix = rating.values.dot(df_ingredient_filtered.values)
    user_preferences = pd.DataFrame(user_recipe_matrix, columns=df_ingredient_filtered.columns, index=rating.index)
    
    print(user_preferences.loc[:, (user_preferences != 0).any(axis=0)])

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

def predict(df_predict, user_no):
    # Load data
    df_food_data = pd.read_json(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", "train.json"))
    df_food_data.set_index('id', inplace=True)

    #user predicted rating to all books
    user_predicted_rating = df_predict[user_no]

    #combine book rating and book detail
    user_rating_food = pd.concat([user_predicted_rating,df_food_data], axis=1)

    return user_rating_food.sort_values(by=[user_no], ascending=False).iloc[0:10]

if __name__ == "__main__":
    train()

    df_predict = pd.read_pickle(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations", "data", 'tfidf_model.pkl'))
    print(predict(df_predict, 1))