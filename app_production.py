import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
from flask import Flask, request, jsonify, render_template # render_template serves the HTML
# Import any other necessary sklearn modules if needed for helper functions
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler etc. (if needed directly here)
from tensorflow.keras.preprocessing.sequence import pad_sequences # Needed if regenerating sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json # Needed if regenerating sequences

# ==============================================================================
#  Load Artifacts ONCE when the application starts
# ==============================================================================
print("Loading application artifacts...")
try:
    model = keras.models.load_model('movie_recommendation_model.h5')
    scaler = joblib.load('scaler.pkl')
    min_max_scaler = joblib.load('min_max_scaler.pkl')
    director_encoder = joblib.load('director_encoder.pkl')
    #title_tokenizer = joblib.load('title_tokenizer.pkl')

    print("Loading tokenizer from JSON...")
    try: # Add try/except around file loading
        with open('title_tokenizer.json', 'r', encoding='utf-8') as f:
            # Load the outer JSON structure (which contains the config string)
            tokenizer_json_str = json.load(f)
            # Recreate the tokenizer object from the configuration string
            title_tokenizer = tokenizer_from_json(tokenizer_json_str)
        print("Tokenizer loaded successfully from JSON.")
    except FileNotFoundError:
        print("ERROR: title_tokenizer.json not found!")
        # Handle error appropriately - maybe raise it to stop startup
        raise
    except Exception as e:
        print(f"ERROR loading tokenizer from JSON: {e}")
        raise # Reraise to see the error in logs

    with open('actor_to_index.json', 'r') as f:
        actor_to_index = json.load(f)
    all_actors = np.load('all_actors.npy', allow_pickle=True).tolist()
    with open('genre_columns.json', 'r') as f:
        genre_columns = json.load(f) # Needed for process_preferences & apply_filtering

    # Load preprocessed movie data
    # Use the final preprocessed CSV that includes necessary columns
    movies_df = pd.read_csv("neural_net_ready_preprocessed_movies.csv") # Make sure path is correct

    # --- Recreate/Load necessary feature columns for prediction ---
    # (Do this once at startup)
    if 'Director_Encoded' not in movies_df.columns:
         movies_df['Director_Encoded'] = director_encoder.transform(movies_df['Director'])

    # Assuming title_data (padded_sequences) is needed and not in CSV
    sequences = title_tokenizer.texts_to_sequences(movies_df['Title'])
    all_title_data = pad_sequences(sequences, maxlen=20) # Assign to variable

    # Assuming star_cast_encoded is needed and not in CSV
    # Ensure 'Star Cast List Clean' column exists or recreate it first
    if 'Star Cast List Clean' not in movies_df.columns:
        # You might need more preprocessing steps here if this column isn't saved
        raise ValueError("Column 'Star Cast List Clean' not found in preprocessed CSV.")
        # If it is saved but as a string:
        # movies_df['Star Cast List Clean'] = movies_df['Star Cast List Clean'].apply(eval)

    all_star_cast_data = np.zeros((len(movies_df), len(all_actors)))
    for i, actors in enumerate(movies_df['Star Cast List Clean']):
         # Handle cases where actors might not be a list (e.g., NaN or needs eval)
         if isinstance(actors, list):
             for actor in actors:
                 if actor in actor_to_index:
                     index = actor_to_index[actor]
                     all_star_cast_data[i, index] = 1

    # Prepare other features
    all_numerical_data = movies_df[['IMDb Rating', 'Duration (minutes)', 'Year']].values
    all_director_data = movies_df['Director_Encoded'].values
    all_genre_data = movies_df[genre_columns].values # Use loaded genre columns

    num_movies = len(movies_df)

    # --- Calculate Base Predictions using Generic User ID (ONCE) ---
    print("Calculating base predictions...")
    generic_user_id = 0
    generic_user_array = np.array([generic_user_id] * num_movies)
    movie_id_array = np.arange(num_movies) # Assuming movie embedding uses index

    # Ensure features are in the correct order for model.predict
    base_predictions = model.predict([
        generic_user_array,
        movie_id_array,
        all_numerical_data,
        all_director_data,
        all_genre_data,
        all_star_cast_data,
        all_title_data
    ])
    # Add base score to the main DataFrame - use a consistent name like 'base_score'
    movies_df['base_score'] = base_predictions.flatten()
    print("Base predictions calculated and added.")

    print("Artifacts loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR during artifact loading: {e}")
    # Handle error appropriately - maybe exit or disable the /recommend route
    raise e # Reraise to stop app if loading fails


# ==============================================================================
#  Helper Functions (Processing & Filtering)
# ==============================================================================

def process_preferences(raw_prefs_dict, all_genre_columns):
    """Converts the raw dictionary from JSON into a structured dictionary."""
    # Raw data comes from request.get_json(), which is already a dict
    processed = {}

    # **Important**: Keys here must match keys sent by JavaScript ('What Genres...' etc.)
    # and the structure expected by apply_filtering.
    raw_genres_str = raw_prefs_dict.get('What Genres do you enjoy?', '')
    processed['genres'] = [g.strip() for g in raw_genres_str.split(',') if g.strip()]

    processed['era'] = raw_prefs_dict.get('Do you prefer older classics or newer releases?', '')
    processed['length'] = raw_prefs_dict.get('Do you prefer shorter movies or longer epics?', '')

    # Assuming JS sends combined actors/directors string in this key
    raw_actors_directors = raw_prefs_dict.get('Are there any actors or directors your particularly enjoy?', '')
    if isinstance(raw_actors_directors, str) and raw_actors_directors.lower().strip() not in ['no', 'yes', 'nope', '']:
        processed['actors_directors'] = [name.strip() for name in raw_actors_directors.split(',') if name.strip()]
    else:
         processed['actors_directors'] = []

    processed['acclaim'] = raw_prefs_dict.get('Do you prefer critically acclaimed movies or more popular ones?', '') # Add if needed

    print(f"Processed preferences: {processed}") # Debug log
    return processed

def apply_filtering(movies_with_scores_df, processed_prefs):
    """Filters the movie DataFrame based on user preferences."""
    filtered_df = movies_with_scores_df.copy()
    print(f"Initial rows before filtering: {len(filtered_df)}")

    # Filter by Genre
    if processed_prefs.get('genres'):
         valid_genres = [g for g in processed_prefs['genres'] if g in filtered_df.columns]
         if valid_genres:
             print(f"Filtering by genres: {valid_genres}")
             genre_filter = filtered_df[valid_genres].sum(axis=1) > 0
             filtered_df = filtered_df[genre_filter]
             print(f"Rows after genre filter: {len(filtered_df)}")
         else:
             print("No valid genres selected or found in columns.")

    # === Add Filtering Logic for Era and Length ===
    # These require the original (non-scaled) Year and Duration columns
    # Ensure these columns exist in your movies_df (add them during preprocessing/loading if needed)

    # Filter by Era
    era_pref = processed_prefs.get('era', '')
    if 'Original_Year' in filtered_df.columns:
        print(f"Filtering by era: {era_pref}")
        if 'Post-2000' in era_pref:
            filtered_df = filtered_df[filtered_df['Original_Year'] > 2000]
        elif 'Pre-1980' in era_pref:
            filtered_df = filtered_df[filtered_df['Original_Year'] < 1980]
        elif '1980-2000' in era_pref:
            filtered_df = filtered_df[(filtered_df['Original_Year'] >= 1980) & (filtered_df['Original_Year'] <= 2000)]
        # If 'No Preference', don't filter
        print(f"Rows after era filter: {len(filtered_df)}")
    elif era_pref and 'No Preference' not in era_pref:
         print("WARNING: Era preference provided, but 'Original_Year' column not found for filtering.")


    # Filter by Length
    length_pref = processed_prefs.get('length', '')
    if 'Original_Duration' in filtered_df.columns:
        print(f"Filtering by length: {length_pref}")
        if 'Shorter' in length_pref:
            filtered_df = filtered_df[filtered_df['Original_Duration'] <= 100]
        elif 'Longer' in length_pref:
            filtered_df = filtered_df[filtered_df['Original_Duration'] >= 120]
        # If 'No Preference', don't filter
        print(f"Rows after length filter: {len(filtered_df)}")
    elif length_pref and 'No Preference' not in length_pref:
         print("WARNING: Length preference provided, but 'Original_Duration' column not found for filtering.")


    # Add other filters if needed (e.g., based on 'actors_directors' list)

    print(f"Final rows after all filtering: {len(filtered_df)}")
    return filtered_df


# ==============================================================================
#  Flask Application Setup
# ==============================================================================
app = Flask(__name__)

# --- Route to serve the main HTML page ---
@app.route('/')
def home():
    # Assumes index.html is in a 'templates' folder sibling to app.py
    # Or just in the same directory if using Flask default static serving for simple cases
    return render_template('index.html')

# --- Route to handle recommendation requests ---
@app.route('/recommend', methods=['POST'])
def recommend_route(): # Renamed function slightly to avoid clash with any variable
    try:
        # 1. Get preference data from request body
        raw_prefs = request.get_json()
        if not raw_prefs:
            print("Received empty request data.")
            return jsonify({"error": "No preference data received"}), 400
        print(f"Received raw preferences: {raw_prefs}")

        # 2. Process the raw preferences
        processed_prefs = process_preferences(raw_prefs, genre_columns)

        # 3. Apply Filtering (using the movies_df with pre-calculated base_score)
        filtered_recommendations = apply_filtering(movies_df, processed_prefs)

        # 4. Sort filtered results and get Top N
        # Make sure to sort by the column containing the base scores ('base_score')
        final_recommendations = filtered_recommendations.sort_values(by='base_score', ascending=False)
        top_n = 10
        # Return the 'base_score' used for ranking
        user_top_movies = final_recommendations.head(top_n)[['Title', 'base_score']].to_dict(orient='records')

        print(f"Sending {len(user_top_movies)} recommendations.")
        # 5. Return results as JSON
        return jsonify({"recommendations": user_top_movies})

    except Exception as e:
        # Log the detailed error on the server for debugging
        print(f"ERROR during /recommend processing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        # Return a generic error message to the client
        return jsonify({"error": "An internal error occurred while generating recommendations."}), 500

# ==============================================================================
#  Run the Flask App
# ==============================================================================
if __name__ == '__main__':
    print("Starting Flask application...")
    # Ensure host='0.0.0.0' if running in Docker or needing external access
    # debug=True is helpful for development, shows errors in browser, auto-reloads
    # Set debug=False for production deployment
    app.run(debug=False)