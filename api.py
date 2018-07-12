__author__ = 'VladimirSveshnikov'

import gensim
from scipy.spatial.distance import euclidean
import string
import inflect

from flask import Flask, request, jsonify


app = Flask(__name__)


print('start app')
# model link: https://fasttext.cc/docs/en/english-vectors.html
model_path = '/mnt/data/crawl-300d-2M.vec'
model_fastText = gensim.models.KeyedVectors.load_word2vec_format(model_path)
# other way:
# from gensim.models.wrappers import FastText
# model = FastText.load_fasttext_format('wiki.simple')

print('model loaded')
p = inflect.engine()


def clean_str(s):
    for char in string.punctuation:
        s = s.replace(char, "")
    s = s.split()
    for count in range(len(s)):
        if s[count].isdigit():
            s[count] = p.number_to_words(int(s[count]))
    return ' '.join(s)


def get_similarity_euql(model, first_sentence, second_sentence):
    similarity = 0
    if first_sentence == second_sentence:
        return 0.0

    first_sentence = [i for i in clean_str(first_sentence).split() if i in model]
    second_sentence = [i for i in clean_str(second_sentence).split() if i in model]
    
    if len(first_sentence) == 0:
        return 's1 sentence is too short or the model does not contain a word from this sentence'

    if len(second_sentence) == 0:
        return 's2 sentence is too short or the model does not contain a word from this sentence'

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
        s1 = clean_str(request.args.get('s1').lower())
        s2 = clean_str(request.args.get('s2').lower())
        dist = get_similarity_euql(model_fastText, s1, s2)

        return jsonify({'dist': dist})
    else:
        return 'you should use GET method'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
