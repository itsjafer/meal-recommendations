import pandas as pd 
import numpy as np
import os
import pickle
import csv
from sklearn.preprocessing import MultiLabelBinarizer

pd.options.display.max_colwidth = 150

def predict(df_predict, df_food_data, user_no):
    """ Predict the top 10 recipes for a user
    
    Arguments:
        df_predict {DataFrame} -- the Model for user preferences 
        user_no {int} -- the user to predict for
    
    Returns:
        DataFrame -- A list of recipes
    """
    print('Predicting for user id, ' + str(user_no))

    #user predicted rating to all books
    user_predicted_rating = df_predict[int(user_no)]

    #combine book rating and book detail
    user_rating_food = pd.concat([user_predicted_rating,df_food_data], axis=1)

    print('Done predicting!\n')
    return user_rating_food.sort_values(by=[user_no], ascending=False).iloc[0:10]


def cold_start(sample_data, df_food_data, user_no):
    """ In content-based models, the cold start problem are usually solved in one of the following ways:
            1. The user is recommended the most popular items 
            2. The user is recommended things based on other metadata (program, location)
            3. The user is given several recipes and asked to rate them
        For this implementation, we will choose option 3.
    """

    df_user_ratings = pd.DataFrame(columns=['user_id', 'food_id', 'rating'])
    print("\nPlease answer a few questions.")
    for index, row in sample_data.iterrows():
        print(row)
        rating = input("\nWhat do you think of the recipe above? 1 = like, 0 = neutral, -1 = dislike\n")
        df_user_ratings = df_user_ratings.append({'rating': int(rating), 'food_id': int(index), 'user_id':int(user_no)}, ignore_index=True)

    df_predict = train(df_food_data, df_user_ratings, int(user_no))

    return predict(df_predict, df_food_data, user_no)


def train(df_food_data, user_pref_df, user_id):
    """This will update the parameter values for a user based on a new recipe rating
    
    Arguments:
        user_id {int} -- The id of the user whose preference to update
        recipe_id {int} -- The id of the recipe that is being rated
        rating {int} -- either -1, 0, or 1
    
    Returns:
        DataFrame -- The updated model with all preferences
    """
    print("Updating model...")

    # One Hot Encoding
    df_ingredient = df_food_data['ingredients']
    mlb = MultiLabelBinarizer()
    df_ingredient = pd.DataFrame(mlb.fit_transform(df_ingredient),columns=mlb.classes_, index=df_food_data.index)

    # Now we will update the user preferences and save it
    df_user_ratings = pd.DataFrame(columns=['user_id', 'food_id', 'rating'])
    if (os.path.exists(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                        "content_based_meal_recommendations_poc", "data", "user_preferences.csv"))):
        df_user_ratings = pd.read_csv(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                      "content_based_meal_recommendations_poc", "data", "user_preferences.csv"))

    updated_df = df_user_ratings.append(user_pref_df)

    updated_df.to_csv(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                    "content_based_meal_recommendations_poc", "data", 'user_preferences.csv'), index=False)

    user_pref_df = updated_df.loc[updated_df['user_id'] == user_id]
    user_pref_df = user_pref_df.astype(int)
    rating = pd.pivot_table(user_pref_df, values='rating', index=['food_id'], columns = ['user_id']).T
    rating.fillna(0, inplace=True)

    # We find the user's preferences for each ingredient (create user_profile)
    df_ingredient_filtered = df_ingredient[df_ingredient.index.isin(rating.columns.values)].sort_index()

    user_recipe_matrix = rating.values.dot(df_ingredient_filtered.values)
    user_preferences = pd.DataFrame(user_recipe_matrix, columns=df_ingredient_filtered.columns, index=rating.index)
    
    # Calculate IDF
    document_frequency = df_ingredient.sum()
    idf = (len(df_food_data)/document_frequency).apply(np.log)
    idf_df_item = df_ingredient.mul(idf.values)

    # Find the recipes that most match the user's preferences
    working_df = idf_df_item.mul(user_preferences.iloc[0], axis=1)
    df_predict = working_df.sum(axis=1)

    # If the model doesn't yet exist, we will create it
    model = pd.DataFrame()
    if (os.path.exists(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                        "content_based_meal_recommendations_poc", "data", "tfidf_model.pkl"))):
        model = pd.read_pickle(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                "content_based_meal_recommendations_poc", "data", 'tfidf_model.pkl'))
    
    model[int(user_id)] = df_predict
    
    with open(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                           "content_based_meal_recommendations_poc", "data", 'tfidf_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print('Finished training model!\n')
    return model.copy()

if __name__ == "__main__":

    # Load data
    df_food_data = pd.read_json(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations_poc", "data", "train.json"))
    df_food_data.set_index('id', inplace=True)
    df_food_data = df_food_data.head(1000)

    # Get preferences
    if (not os.path.exists(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations_poc", "data", "user_preferences.csv"))):
        user_id = input("\nPlease enter your user id\n")
        print(cold_start(df_food_data.sample(n=3), df_food_data, int(user_id)))

    while True:
        # Load the model (in case changes have been made)
        df_predict = pd.read_pickle(os.path.join(os.getenv("DATA_SCIENCE_DIR"), 
                                        "content_based_meal_recommendations_poc", "data", 'tfidf_model.pkl'))

        user_id = input("\nPlease enter your user id\n")
        if user_id.isnumeric():
            # Check if the user id exists
            if (int(user_id) not in df_predict.columns.values):
                # Ask cold start questions and update model online
                print(cold_start(df_food_data.sample(n=3), df_food_data, int(user_id)))
                continue

            # If it exists, ask if they want to rate some more recipes or just get predictions
            if (input("Enter Y if you'd like to rate some more recipes. Press anything else to recieve your recommendations\n") == 'Y'):
                print(cold_start(df_food_data.sample(n=3), df_food_data, int(user_id)))
                continue
            
            print(predict(df_predict, df_food_data, int(user_id)))
        else:
            break