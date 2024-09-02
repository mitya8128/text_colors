import gensim.downloader as api
import numpy as np
from scipy.spatial.distance import cosine
import logging
import re

# Set up logging to output to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the pre-trained Word2Vec model
logger.info("Loading the Word2Vec model...")
model = api.load('word2vec-google-news-300')
logger.info("Word2Vec model loaded successfully.")

# Define a list of color names
color_names = [
    'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
    'black', 'white', 'gray', 'orange', 'purple', 'pink', 'brown'
]

# Convert color names to Word2Vec vectors, handling case sensitivity
logger.info("Converting color names to Word2Vec vectors...")
color_vectors = {}
for color in color_names:
    try:
        color_vectors[color] = model[color]
    except KeyError:
        logger.warning(f"Color '{color}' not found in the Word2Vec model vocabulary.")

logger.info("Color names successfully converted to vectors.")

# Function to find the nearest color based on cosine similarity
def nearest_color(word_vector, color_vectors):
    min_distance = float('inf')
    closest_color = None
    
    for color, color_vector in color_vectors.items():
        distance = cosine(word_vector, color_vector)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
            
    return closest_color, min_distance

# Function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Split text into words
    words = text.split()
    return words

# Sample input text
input_text = "The sky is blue and the grass is green. The sun is bright yellow."

logger.info("Processing the input text.")
# Preprocess text
words = preprocess_text(input_text)
logger.info(f"Preprocessed text: {' '.join(words)}")

# Filter out words not in the model's vocabulary
words_in_vocab = []
for word in words:
    if word in model.key_to_index:
        words_in_vocab.append(word)
    else:
        logger.warning(f"Word '{word}' not found in the Word2Vec model vocabulary.")

logger.info(f"Filtered {len(words_in_vocab)} words from the input text that are in the model's vocabulary.")

# Map words to their nearest colors and log the results
logger.info("Mapping words to their nearest colors...")
word_colors = []
for word in words_in_vocab:
    word_vector = model[word]
    closest_color, distance = nearest_color(word_vector, color_vectors)
    word_colors.append(closest_color)
    logger.info(f"Word '{word}' is closest to color '{closest_color}' with distance {distance:.4f}")

# Count frequency of each color
color_counts = {}
for color in word_colors:
    color_counts[color] = color_counts.get(color, 0) + 1

logger.info("Color frequencies calculated.")
# Print color frequencies
print("Color frequencies:")
print(color_counts)

# Average all word vectors to get one color vector
logger.info("Calculating the average vector of all words.")
average_vector = np.mean([model[word] for word in words_in_vocab], axis=0)

# Find the nearest color for the average vector
average_color, average_distance = nearest_color(average_vector, color_vectors)
logger.info(f"Calculated the nearest color for the average vector: {average_color} with distance {average_distance:.4f}")

print("\nThe average color for the text is:", average_color)

