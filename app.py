# taken from https://community.canvaslms.com/thread/2595

from flask import Flask, render_template,url_for, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import re
import io
import base64
from keras.models import load_model
app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def chr_demo():
    guess = "_"
    if request.method== 'POST':
        #requests image from url 
        img_size = 28, 28 
        image_url = request.values['imageBase64']  
        image_string = re.search(r'base64,(.*)', image_url).group(1)  
        image_bytes = io.BytesIO(base64.b64decode(image_string)) 
        image = Image.open(image_bytes) 
        image = image.resize(img_size, Image.LANCZOS)  
        image = image.convert('1')
        #image.save("geeks.jpg")
        image_array = np.asarray(image)
        image_array = np.reshape(image_array, (1, 28, 28, 1))
        #image_array = image_array.astype("float32")
        #image_array /= 255
        with graph.as_default():
            prediction = chr_model.predict(image_array)[0]
            prediction = np.argmax(prediction)
            guess = str(letters[int(prediction)+1])
            return jsonify(guess = guess) #returns as json format

    return render_template('index.html', guess = guess)

@app.route('/digits_demo', methods=['GET','POST'])
def digits_demo():
    guess = 0
    if request.method== 'POST':
        #requests image from url
        img_size = 28, 28
        image_url = request.values['imageBase64']
        image_string = re.search(r'base64,(.*)', image_url).group(1)
        image_bytes = io.BytesIO(base64.b64decode(image_string))
        image = Image.open(image_bytes)
        image = image.resize(img_size, Image.LANCZOS)
        image = image.convert('1')
        #image.save("geeks.jpg")
        image_array = np.asarray(image)
        image_array = np.reshape(image_array, (1, 28, 28, 1))
        #image_array = image_array.astype("float32")
        #image_array /= 255
        with graph.as_default():
            prediction = digits_model.predict(image_array)[0]
            prediction = np.argmax(prediction)
            guess = int(prediction)
            return jsonify(guess = guess) #returns as json format

    return render_template('digits_demo.html', guess = guess)

if __name__ == '__main__':
    chr_model = load_model('models/emnist_cnn_model.h5')
    digits_model = load_model('models/mnist_digits_cnn.h5')
    graph = tf.get_default_graph()
    letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
        11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
        21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}
    app.run(debug = True)
