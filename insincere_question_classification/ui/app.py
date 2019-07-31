import flask
from keras.models import load_model
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import CustomObjectScope
from custom_object_scope import Attention, f1

app = flask.Flask(__name__)


def init():
    global model, tokenizer, graph
    # load the pre-trained Keras model
    with CustomObjectScope({"Attention": Attention, "f1": f1}):
        model = load_model("insincere_question_classification/models/trainedModel.h5")
    with open(
        "insincere_question_classification/models/tokenizer.pickle", "rb"
    ) as handle:
        tokenizer = pickle.load(handle)

    graph = tf.get_default_graph()


def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET")
    response.headers.add(
        "Access-Control-Allow-Headers",
        "accept,content-type,Origin,X-Requested-With,Content-Type,access_token,\
            Accept,Authorization,source",
    )
    response.headers.add("Access-Control-Allow-Credentials", True)
    return response


# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    question = flask.request.args.get("question")
    (raw_prediction, prediction) = predict_question(question)
    return sendResponse(
        {"Question": question, "Class": prediction, "Probability": str(raw_prediction)}
    )


@app.route("/response", methods=["GET"])
def index():
    question = flask.request.args.get("question")
    (raw_prediction, prediction) = predict_question(question)
    return flask.render_template(
        "index.html",
        question=question,
        prediction=prediction,
        probability=raw_prediction,
    )


@app.route("/", methods=["GET"])
def render_main():
    return flask.render_template("index.html")


def predict_question(question):
    text = tokenizer.texts_to_sequences([question])
    question_value = pad_sequences(text, maxlen=65)
    with graph.as_default():
        raw_prediction = model.predict(question_value)[0][0]
    if raw_prediction < 0.44:
        prediction = "Sincere"
    else:
        prediction = "Insincere"
    return (raw_prediction, prediction)


if __name__ == "__main__":
    print(
        (
            "* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"
        )
    )
    init()
    app.run(host="0.0.0.0", threaded=True)
