__author__ = 'VladimirSveshnikov'

import gensim
from scipy.spatial.distance import euclidean
import string

from flask import Flask, request, jsonify


app = Flask(__name__)


print('start app')
# model link: https://fasttext.cc/docs/en/english-vectors.html
model_path = 'your_path_to_model'
model_fastText = gensim.models.KeyedVectors.load_word2vec_format(model_path)
print('model loaded')


def clean_str(s):
    for c in string.punctuation:
        s = s.replace(c, "")
    return s


def get_similarity_euql(model, first_sentence, second_sentence):
    similarity = 0
    first_sentence = [i for i in clean_str(first_sentence).split() if i in model]
    second_sentence = [i for i in clean_str(second_sentence).split() if i in model]
    for i in first_sentence:
        first_vector = model[i]
        sim_i = 0
        for j in second_sentence:
            second_vector = model[j]
            sim_i += euclidean(first_vector, second_vector)
        similarity += sim_i / len(second_sentence)
    return similarity / len(first_sentence)


@app.route("/get_dist", methods=["GET"])
def api():
    if request.method == "GET":
        s1 = request.args.get('s1')
        s2 = request.args.get('s2')
        dist = get_similarity_euql(model_fastText, s1, s2)

        return jsonify({'dist': dist})
    else:
        return 'you should use GET method'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
