## What it does
Our model converts American Sign Language Alphabet to English alphabet through a live video feed. Once a word is formed, if there is a typo, the model autocorrects it to the right word.
## How we built it
Through TensorFlow's Keras, we trained a Sequential model on a self-created dataset of 500 images per ASL letter, resulting in an accuracy of around 98%. We then integrated our model through the Flask web framework to predict the letters, and consequently words, through a live video-feed. The user can see the letter and word predictions at the bottom of the screen.
## Challenges we ran into
One of the biggest challenges we had to overcome was the integration of our machine learning model to our live feed as the image dimensions were conflicting. We resolved this by experimenting through different platforms such as flutter, javascript and finally, python.
## Accomplishments that we're proud of
We are proud that we were able to train an accurate model to predict the letters and words in a small amount of time and we hope to incorporate sentence prediction soon to our application.
## What's next for ASL
In the future, we want to expand this to predict and autocomplete sentences based on just a few words. Implementing that would require training a model with huge amounts of data for each word. 
# HackMIT
