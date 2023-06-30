import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from scipy import stats
import pandas as pd
import os
import math


def plot_similarity(labels, features, rotation, version):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title(version)

def cosine_similarity(vector_1, vector_2):
    """
    Compute cosine similarity between two vectors.

    Args:
        vector_1 (List[float]): A list of float values representing the first vector.
        vector_2 (List[float]): A list of float values representing the second vector.

    Returns:
        float: The cosine similarity between the two vectors, which is the dot product of
        the two vectors divided by the product of their magnitudes.
    """
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(vector_1)):
        x = vector_1[i]
        y = vector_2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def sts_benchmark(model, type="dev"):
    """
    Compute the Pearson correlation between predicted cosine similarity scores and human-labeled similarity scores
    on the STS benchmark dataset.

    Args:
        model (tensorflow.keras.Model): A trained sentence embedding model that takes in an input sentence and
                                        outputs a corresponding sentence embedding.
        type (str): The type of STS benchmark dataset to use. Either "dev" for the development dataset or "test"
                    for the test dataset. Default is "dev".

    Returns:
        tuple: A tuple containing the Pearson correlation coefficient and the p-value of the correlation test.
    """

    def _get_sts_dataset(type="test"):
        """

        :param type:
        :return:
        """

        sts_dataset = tf.keras.utils.get_file(
            fname="Stsbenchmark.tar.gz",
            origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
            extract=True,
        )
        if type == "dev":
            data = pd.read_table(
                os.path.join(
                    os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"
                ),
                on_bad_lines="skip",
                engine="python",
                skip_blank_lines=True,
                usecols=[4, 5, 6],
                names=["sim", "sent_1", "sent_2"],
            )
        else:
            data = pd.read_table(
                os.path.join(
                    os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"
                ),
                on_bad_lines="skip",
                engine="python",
                skip_blank_lines=True,
                usecols=[4, 5, 6],
                names=["sim", "sent_1", "sent_2"],
            )

        return data

    data = _get_sts_dataset(type=type)
    data = data[[isinstance(s, str) for s in data["sent_2"]]].reset_index()

    # prepare data
    base_text = [data["sent_1"][i] for i in range(len(data))]
    ref_text = [data["sent_2"][i] for i in range(len(data))]
    scores = data["sim"].tolist()

    base_vectors = []
    ref_vectors = []

    # get text vectors from base, tuned, pre-tuned models
    for i in range(len(base_text)):
        base_vectors.append(list(model.predict([base_text[i]])[0]))
        ref_vectors.append(list(model.predict([ref_text[i]])[0]))

    base_cosine_similarity = [
        cosine_similarity(base_vectors[i], ref_vectors[i])
        for i in range(len(base_text))
    ]
    return stats.pearsonr(scores, base_cosine_similarity)

def process_model_input(data):
    """
    Processes the input data by reshaping the left and right inputs and converting the similarity values.

    Args:
        data: (dict) A dictionary containing the keys "base", "ref", and "similarity",
            with values corresponding to the input base text, reference text, and similarity values respectively.

    Returns:
        tuple: A tuple of three numpy arrays containing the preprocessed left inputs, right inputs,
            and similarity values respectively.
    """
    text_list = [list(data["base"].values), list(data["ref"].values)]
    left_inputs = np.asarray(text_list[0])
    right_inputs = np.asarray(text_list[1])
    left_inputs = left_inputs.reshape(
        left_inputs.shape[0],
    )
    right_inputs = right_inputs.reshape(
        right_inputs.shape[0],
    )

    # 1 if we inputs are semantically similiar, 0 if not.
    # Check the distance function defined as 1-arccos(similiarity)/pi which has range between 1,0 for domain 0 to 1
    similarity = np.asarray(list(data["similarity"].values))

    return left_inputs, right_inputs, similarity

