import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
from flask import Flask, request, jsonify, render_template # render_template serves the HTML
# Import any other necessary sklearn modules if needed for helper functions
# from sklearn.preprocessing import LabelEncoder # Not used directly here anymore
from tensorflow.keras.preprocessing.sequence import pad_sequences # Needed if regenerating sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json # Needed if regenerating sequences

# ==============================================================================
#  Load Artifacts ONCE when the application starts
# ==============================================================================
print("Loading application artifacts...")
try:
    # --- Load Model and Preprocessors ---
    # Use the model trained on the simulated data
    model = keras.models.load_model('movie_recommendation_model.h5') #<-- Make sure this is your latest model
    scaler = joblib.load('scaler.pkl') # Assuming StandardScaler was used
    director_encoder = joblib.load('director_encoder.pkl')

    print("Loading tokenizer from JSON...")
    try:
        with open('title_tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_json_str = json.load(f)
            title_tokenizer = tokenizer_from_json(tokenizer_json_str)
        print("Tokenizer loaded successfully from JSON.")
    except FileNotFoundError:
        print("ERROR: title_tokenizer.json not found!")
        raise
    except Exception as e:
        print(f"ERROR loading tokenizer from JSON: {e}")
        raise

    with open('actor_to_index.json', 'r') as f:
        actor_to_index = json.load(f)
    all_actors = np.load('all_actors.npy', allow_pickle=True).tolist()
    with open('genre_columns.json', 'r') as f:
        genre_columns = json.load(f) # Needed for process_preferences & apply_filtering

    # --- Load and Prepare Movie Data ---
    # Load the preprocessed CSV that includes ORIGINAL columns for filtering
    # Make sure this CSV was generated by the preprocessing script that keeps originals
    movies_df = pd.read_csv("neural_net_ready_preprocessed_movies.csv") # <-- Make sure this has originals
    print(f"Loaded movies_df with columns: {movies_df.columns.tolist()}")

    # Ensure 'Star Cast List Clean' is a list if loaded as string
    if 'Star Cast List Clean' in movies_df.columns and isinstance(movies_df['Star Cast List Clean'].iloc[0], str):
         try:
            print("Attempting to eval 'Star Cast List Clean' column...")
            # Use apply with a lambda to handle potential errors gracefully
            movies_df['Star Cast List Clean'] = movies_df['Star Cast List Clean'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
            print("'Star Cast List Clean' processed.")
         except Exception as e:
             print(f"WARNING: Could not evaluate 'Star Cast List Clean'. Actor filtering might not work. Error: {e}")
             # Consider raising an error or disabling actor filtering if this fails


    # --- CHECK/CREATE Original Columns for Filtering (Crucial!) ---
    # Ensure the columns used by apply_filtering exist. Create if necessary.
    if 'Original_Year' not in movies_df.columns and 'Year_Original' in movies_df.columns:
         # Handle potential naming difference from preprocessing example
         movies_df.rename(columns={'Year_Original': 'Original_Year'}, inplace=True)
    elif 'Original_Year' not in movies_df.columns and 'Year' in movies_df.columns:
         print("WARNING: 'Original_Year' column not found. Attempting to use 'Year' (unscaled) for filtering. Ensure 'Year' holds original values.")
         # This assumes 'Year' somehow holds original values - less robust. Best practice is having *_Original columns.
         # If 'Year' holds SCALED values, filtering won't work correctly!
         # You might need to load/merge original data here if the CSV doesn't have it.
         # For now, we proceed assuming the user ensured originals are available somehow.

    if 'Original_Duration' not in movies_df.columns and 'Duration (minutes)_Original' in movies_df.columns:
         movies_df.rename(columns={'Duration (minutes)_Original': 'Original_Duration'}, inplace=True)
    elif 'Original_Duration' not in movies_df.columns and 'Duration (minutes)' in movies_df.columns:
         print("WARNING: 'Original_Duration' column not found. Attempting to use 'Duration (minutes)' for filtering. Ensure it holds original values.")
         # Same warning as for Year applies here.

    # --- Prepare Feature Arrays for Prediction (Run ONCE at startup) ---
    # (Do this once at startup for the base score calculation)
    if 'Director_Encoded' not in movies_df.columns:
         movies_df['Director_Encoded'] = director_encoder.transform(movies_df['Director'])

    sequences = title_tokenizer.texts_to_sequences(movies_df['Title'].astype(str)) # Ensure Title is string
    all_title_data = pad_sequences(sequences, maxlen=20)

    all_star_cast_data = np.zeros((len(movies_df), len(all_actors)))
    for i, actors in enumerate(movies_df['Star Cast List Clean']):
        if isinstance(actors, list):
            for actor in actors:
                if actor in actor_to_index:
                    index = actor_to_index[actor]
                    all_star_cast_data[i, index] = 1

    # Use SCALED data for prediction input
    # Make sure these column names match exactly what's in your CSV after preprocessing
    all_numerical_data = movies_df[['IMDb Rating', 'Duration (minutes)', 'Year']].values
    all_director_data = movies_df['Director_Encoded'].values
    # Ensure genre_columns list correctly identifies the one-hot columns in movies_df
    all_genre_data = movies_df[genre_columns].values

    num_movies = len(movies_df)

    # --- Calculate Base Predictions using Generic User ID (ONCE) ---
    print("Calculating base predictions for all movies...")
    generic_user_id = 0 # Use a consistent generic ID
    generic_user_array = np.array([generic_user_id] * num_movies)
    movie_id_array = np.arange(num_movies) # Movie IDs are their index in the dataframe

    # Ensure features are in the correct order for model.predict
    base_predictions = model.predict([
        generic_user_array,    # User ID input
        movie_id_array,        # Movie ID input (for movie embedding)
        all_numerical_data,    # Scaled numerical features
        all_director_data,     # Encoded director
        all_genre_data,        # One-hot genres
        all_star_cast_data,    # Multi-hot actors
        all_title_data         # Padded title sequences
    ], batch_size=256) # Use a larger batch size for prediction
    # Add base score to the main DataFrame
    movies_df['base_score'] = base_predictions.flatten()
    print("Base predictions calculated and added to movies_df.")

    print("Artifacts loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR during artifact loading: Missing file {e}")
    raise e
except KeyError as e:
    print(f"FATAL ERROR during artifact loading: Missing expected column {e} in CSV. Ensure preprocessing saved originals and check column names.")
    raise e
except Exception as e:
    print(f"FATAL ERROR during artifact loading: {e}")
    import traceback
    traceback.print_exc()
    raise e


# ==============================================================================
#  Helper Functions (Processing & Filtering) - Keep these as they are
# ==============================================================================

def process_preferences(raw_prefs_dict, all_genre_columns):
    """Converts the raw dictionary from JSON into a structured dictionary."""
    processed = {}
    # Get genres
    raw_genres_str = raw_prefs_dict.get('What Genres do you enjoy?', '')
    # --- Store only valid genres present in the dataframe columns ---
    processed['genres'] = [g.strip() for g in raw_genres_str.split(',') if g.strip() and g.strip() in all_genre_columns]

    # --- Directly use the values received from the form for era and length ---
    processed['era'] = raw_prefs_dict.get('Do you prefer older classics or newer releases?', 'All of the above - no preference')
    processed['length'] = raw_prefs_dict.get('Do you prefer shorter movies or longer epics?', 'All of the above - no preference')
    # --- End Direct Use ---

    # Get actors/directors (combine from separate form fields)
    actors = raw_prefs_dict.get('actors', '')
    directors = raw_prefs_dict.get('directors', '')
    actors_directors_list = []
    if actors:
        actors_directors_list.extend([a.strip().lower() for a in actors.split(',') if a.strip()]) # Use lowercase
    if directors:
        actors_directors_list.extend([d.strip().lower() for d in directors.split(',') if d.strip()]) # Use lowercase

    processed['actors_directors'] = actors_directors_list

    print(f"Processed preferences: {processed}") # Now this should show the correct era/length strings
    return processed

def apply_filtering(movies_to_filter_df, processed_prefs):
    """Filters the movie DataFrame based on user preferences. Uses EXACT column names from debug."""
    filtered_df = movies_to_filter_df.copy()
    print(f"Filtering {len(filtered_df)} recommendations...")

    # Debug: Print available columns (keep this for verification if needed)
    # print(f"\nDEBUG: Columns available INSIDE apply_filtering: {filtered_df.columns.tolist()}\n")

    # Filter by Genre
    if processed_prefs.get('genres'):
        valid_genres = processed_prefs['genres']
        if valid_genres:
            print(f"Filtering by genres: {valid_genres}")
            # Check if ALL genre columns exist before trying to sum
            if all(g in filtered_df.columns for g in valid_genres):
                 genre_filter_mask = filtered_df[valid_genres].sum(axis=1) > 0
                 filtered_df = filtered_df[genre_filter_mask]
                 print(f"-> Count after genre filter: {len(filtered_df)}")
            else:
                 missing_g = [g for g in valid_genres if g not in filtered_df.columns]
                 print(f"WARNING: Skipping genre filter. Missing genre columns: {missing_g}")

    # --- Filter by Era using EXACT column name 'Original_Year' ---
    era_pref = processed_prefs.get('era', '')
    year_col_name = 'Original_Year' # Exact name from debug output
    print(f"DEBUG: Checking for column '{year_col_name}'...")
    if year_col_name in filtered_df.columns:
        print(f"DEBUG: Column '{year_col_name}' FOUND.")
        # Ensure the column is numeric before comparison
        if pd.api.types.is_numeric_dtype(filtered_df[year_col_name]):
            if 'Newer Releases (Post-2000)' in era_pref:
                print(f"Filtering by era: Post-2000")
                filtered_df = filtered_df[filtered_df[year_col_name] > 2000]
            elif 'Older Classics (Pre-1980)' in era_pref:
                print(f"Filtering by era: Pre-1980")
                filtered_df = filtered_df[filtered_df[year_col_name] < 1980]
            elif 'In Between (1980-2000)' in era_pref:
                print(f"Filtering by era: 1980-2000")
                filtered_df = filtered_df[(filtered_df[year_col_name] >= 1980) & (filtered_df[year_col_name] <= 2000)]
            print(f"-> Count after era filter: {len(filtered_df)}")
        else:
            print(f"WARNING: Column '{year_col_name}' is not numeric. Skipping era filter.")
    elif era_pref and 'All of the above' not in era_pref:
         print(f"WARNING: Era preference provided, but '{year_col_name}' column not found for filtering.")

    # --- Filter by Length using EXACT column name 'Original_Duration' ---
    length_pref = processed_prefs.get('length', '')
    duration_col_name = 'Original_Duration' # Exact name from debug output
    print(f"DEBUG: Checking for column '{duration_col_name}'...")
    if duration_col_name in filtered_df.columns:
        print(f"DEBUG: Column '{duration_col_name}' FOUND.")
         # Ensure the column is numeric before comparison
        if pd.api.types.is_numeric_dtype(filtered_df[duration_col_name]):
            if 'Shorter movies (~100 minutes or less)' in length_pref:
                print(f"Filtering by length: Shorter <= 100")
                filtered_df = filtered_df[filtered_df[duration_col_name] <= 100]
            elif 'Longer epics (~120 minutes or longer)' in length_pref:
                print(f"Filtering by length: Longer >= 120")
                filtered_df = filtered_df[filtered_df[duration_col_name] >= 120]
            print(f"-> Count after length filter: {len(filtered_df)}")
        else:
             print(f"WARNING: Column '{duration_col_name}' is not numeric. Skipping length filter.")
    elif length_pref and 'All of the above' not in length_pref:
         print(f"WARNING: Length preference provided, but '{duration_col_name}' column not found for filtering.")

    # Filter by Actors/Directors (Optional)
    # ... (rest of actor/director filtering code - ensure it uses correct columns like 'Director', 'Star Cast List Clean') ...

    print(f"-> Final count after all filtering: {len(filtered_df)}")
    return filtered_df


# ==============================================================================
#  Flask Application Setup
# ==============================================================================
app = Flask(__name__)

# --- Route to serve the main HTML page ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Route to handle recommendation requests ---
@app.route('/recommend', methods=['POST'])
def recommend_route():
    try:
        # 1. Get preference data from request body
        raw_prefs = request.get_json()
        if not raw_prefs:
            print("Received empty request data.")
            return jsonify({"error": "No preference data received"}), 400
        print(f"Received raw preferences: {raw_prefs}")

        # 2. Process the raw preferences into a structured dict
        processed_prefs = process_preferences(raw_prefs, genre_columns)

        # 3. Get Initial Top K Recommendations based on pre-calculated base_score
        #    (movies_df is already sorted implicitly by index, but base_score is what matters)
        k = 200 # Number of initial candidates to consider
        top_k_recommendations = movies_df.sort_values(by='base_score', ascending=False).head(k)
        print(f"Considering top {k} movies based on base score.")

        # 4. Apply Filtering to the Top K list using the helper function
        #    This function now takes the top_k dataframe as input
        filtered_recommendations = apply_filtering(top_k_recommendations, processed_prefs)

        # 5. Get Top N from the FILTERED results
        top_n = 10
        # The dataframe is already sorted by 'base_score' from step 3
        final_recommendations = filtered_recommendations.head(top_n)

        # 6. Format results (e.g., just titles)
        # user_top_movies = final_recommendations[['Title', 'base_score']].to_dict(orient='records') # Optionally include score
        user_top_movies = final_recommendations[['Title']].to_dict(orient='records') # Just titles

        print(f"Sending {len(user_top_movies)} recommendations after filtering.")
        # 7. Return results as JSON
        return jsonify({"recommendations": user_top_movies})

    except Exception as e:
        print(f"ERROR during /recommend processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while generating recommendations."}), 500

# ==============================================================================
#  Run the Flask App
# ==============================================================================
if __name__ == '__main__':
    print("Starting Flask application...")
    # Use debug=True only for local development
    # Use host='0.0.0.0' if running in Docker or needs external access
    app.run(debug=False, host='0.0.0.0') # Example for Render deployment