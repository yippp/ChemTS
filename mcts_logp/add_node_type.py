from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
import sascorer
import networkx as nx
from rdkit.Chem import rdmolops


def expanded_node(model, state, val):
    all_nodes = []

    position = []
    position.extend(state)
    get_int = []
    for j in range(len(position)):
        get_int.append(val.index(position[j]))

    x = np.reshape(get_int, (1, len(get_int)))
    x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                   padding='post', truncating='pre', value=0.)

    for i in range(30):
        predictions = model.predict(x_pad)
        # print("shape of RNN",predictions.shape)
        preds = np.asarray(predictions[0][len(get_int) - 1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int = np.argmax(next_probas)
        all_nodes.append(next_int)

    all_nodes = list(set(all_nodes))

    # print('all nodes:', all_nodes)

    return all_nodes


def node_to_add(all_nodes, val):
    added_nodes = []
    for i in range(len(all_nodes)):
        added_nodes.append(val[all_nodes[i]])

    # print('added notes: ', added_nodes)

    return added_nodes


def chem_kn_simulation(model, state, val, added_nodes):
    all_posible = []

    end = "\n"
    for i in range(len(added_nodes)):
        position = []
        position.extend(state)
        position.append(added_nodes[i])
        # print(state)
        # print(position)
        # print(len(val2))
        total_generated = []
        get_int = []
        for j in range(len(position)):
            get_int.append(val.index(position[j]))

        x = np.reshape(get_int, (1, len(get_int)))
        x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                       padding='post', truncating='pre', value=0.)
        while not get_int[-1] == val.index(end):
            predictions = model.predict(x_pad)
            # print("shape of RNN",predictions.shape)
            preds = np.asarray(predictions[0][len(get_int) - 1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            # print(predictions[0][len(get_int)-1])
            # print("next probas",next_probas)
            next_int = np.argmax(next_probas)
            get_int.append(next_int)
            x = np.reshape(get_int, (1, len(get_int)))
            x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                           padding='post', truncating='pre', value=0.)
            if len(get_int) > 81:
                break
        total_generated.append(get_int)
        all_posible.extend(total_generated)

    return all_posible


def predict_smile(all_posible, val):
    new_compound = []
    for i in range(len(all_posible)):
        total_generated = all_posible[i]

        generate_smile = []

        for j in range(len(total_generated) - 1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)

    return new_compound


def make_input_smile(generate_smile):
    new_compound = []
    for i in range(len(generate_smile)):
        new_compound.append(''.join(generate_smile[i]))
    # print(new_compound)
    # print(len(new_compound))

    return new_compound


def check_node_type(new_compound, SA_mean, SA_std, logP_mean, logP_std, cycle_mean, cycle_std):
    node_index = []
    valid_compound = []
    score = []

    for i in range(len(new_compound)):
        try:
            m = Chem.MolFromSmiles(str(new_compound[i]))
        except:
            None

        if m != None and len(new_compound[i]) <= 81:
            try:
                logp = Descriptors.MolLogP(m)
            except ValueError:  # habdle Sanitization error: Explicit valence for atom is greater than permitted
                continue
            node_index.append(i)
            valid_compound.append(new_compound[i])
            SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[i]))
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[i]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length
            # print(cycle_score)
            # print(SA_score)
            # print(logp)
            SA_score_norm = (SA_score - SA_mean) / SA_std
            logp_norm = (logp - logP_mean) / logP_std
            cycle_score_norm = (cycle_score - cycle_mean) / cycle_std
            score_one = SA_score_norm + logp_norm + cycle_score_norm
            score.append(score_one)

    return node_index, score, valid_compound
