// script.js

// Get references to the form and the results div
const form = document.getElementById('preference-form');
const resultsDiv = document.getElementById('results');

// Add event listener for form submission
form.addEventListener('submit', async (event) => {
    // Prevent the default form submission behavior
    event.preventDefault();
    console.log("Form submission intercepted by JavaScript.");

    // Show a loading message (optional)
    resultsDiv.innerHTML = '<p>Fetching recommendations...</p>';

    // --- Gather form data ---
    const formData = new FormData(form);
    const preferences = {};

    // Get selected genres (checkboxes)
    preferences.genres = formData.getAll('genre'); // Array of selected values

    // Get selected era (radio button)
    preferences.era = formData.get('era');

    // Get selected length (radio button)
    preferences.length = formData.get('length');

    // Get actors text input
    preferences.actors = formData.get('actors');

    // Get directors text input
    preferences.directors = formData.get('directors');

    // --- Prepare data object to send to backend ---
    // **Important**: The keys in this object MUST match what your
    // Python process_preferences function expects to receive.
    // Based on our previous examples, it likely expects the full questions
    // as keys, similar to the columns in your Google Sheet/DataFrame.

    // Combine actors and directors if your backend expects a single field,
    // otherwise send them separately if the backend handles them separately.
    // Let's combine them for this example, matching the previous Python function.
    let actors_directors_string = '';
    if (preferences.actors && preferences.directors) {
        actors_directors_string = `${preferences.actors}, ${preferences.directors}`;
    } else if (preferences.actors) {
        actors_directors_string = preferences.actors;
    } else if (preferences.directors) {
        actors_directors_string = preferences.directors;
    }

    const dataToSend = {
        // Map the form data to the keys expected by your Python backend
        'What Genres do you enjoy?': preferences.genres.join(','), // Send as comma-separated string
        'Do you prefer older classics or newer releases?': preferences.era,
        'Do you prefer shorter movies or longer epics?': preferences.length,
        'Are there any actors or directors your particularly enjoy?': actors_directors_string, // Combined or separate based on backend needs
        // Add any other keys your backend process_preferences expects, like 'AcclaimPref' if needed
        // 'Do you prefer critically acclaimed movies or more popular ones?': formData.get('acclaim') // Add if you have this input
    };


    // --- Send data to Flask backend using fetch ---
    try {
        const response = await fetch('/recommend', { // Your Flask endpoint
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(dataToSend), // Convert JS object to JSON
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json(); // Parse JSON response from Flask

        // --- Display results ---
        if (result.error) {
            resultsDiv.innerHTML = `<p>Error: ${result.error}</p>`;
        } else if (result.recommendations && result.recommendations.length > 0) {
            let html = '<ul>';
            result.recommendations.forEach(movie => {
                // Adjust key based on what Flask returns ('base_score' or 'predicted_liking')
                const scoreKey = 'base_score' in movie ? 'base_score' : 'predicted_liking';
                const scorePercentage = (movie[scoreKey] * 100).toFixed(1);
                html += `<li>${movie.Title} (Likelihood: ${scorePercentage}%)</li>`;
            });
            html += '</ul>';
            resultsDiv.innerHTML = html;
        } else {
            resultsDiv.innerHTML = '<p>No recommendations found matching your criteria.</p>';
        }

    } catch (error) {
        console.error('Error fetching recommendations:', error);
        resultsDiv.innerHTML = `<p>Failed to fetch recommendations. Error: ${error.message}</p>`;
    }
});