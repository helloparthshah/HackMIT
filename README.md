<h1 align="center"> Bridging the chasm between two worlds</h1>

Currently, about 600,000 people in the United States have some form of hearing impairment. Through personal experiences, we understand the guidance necessary to communicate with a person through ASL. Our software eliminates this and promotes a more connected community - one with a lower barrier entry for sign language users.

Our web-based project detects signs using the live feed from the camera and features like autocorrect and autocomplete reduce the communication time so that the focus is more on communication rather than the modes. Furthermore, the Learn feature enables users to explore and improve their sign language skills in a fun and engaging way. Because of limited time and computing power, we chose to train an ML model on ASL, one of the most popular sign languages - but the extrapolation to other sign languages is easily achievable.

With an extrapolated model, this could be a huge step towards bridging the chasm between the worlds of sign and spoken languages.

# Demo
<iframe width="560" height="315" src="https://www.youtube.com/embed/P52I1pX4JN8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Installation
Clone the source locally:

```sh
$ git clone https://github.com/helloparthshah/HackMIT
$ cd HackMIT
$ cd Flask
```
You'll also need to install
`flask`:

Use your package manager to install `flask`.

```sh
$ pip3 install flask
```

Install project dependencies:

```sh
$ pip3 install tensorflow
$ pip3 install keras
$ pip3 install opencv-python
$ pip3 install autocomplete
```
Start the app:

```sh
$ python3 server.py
```

### Build Flutter version
```sh
$ cd asl
$ flutter build apk
```
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

## License

MIT  © [Parth Shah](http://parthshah.tech)
