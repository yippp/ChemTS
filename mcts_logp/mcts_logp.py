from __future__ import print_function
from math import *
import numpy as np
import random as pr
from load_model import loaded_model
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile, predict_smile, check_node_type, node_to_add, \
    expanded_node


class chemical:

    def __init__(self):
        self.position = ['&']

    def Clone(self):
        st = chemical()
        st.position = self.position[:]
        return st

    def SelectPosition(self, m):
        self.position.append(m)

    def Carbon(self):
        print('Initial state: carbon')
        self.position = ['&', 'C']

    def Benzene(self):
        print('Initial state: benzene')
        self.position = ['&', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1']


class Node:

    def __init__(self, position=None, parent=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.depth = 0

    def Selectnode(self):
        ucb = []
        for i in range(len(self.childNodes)):
            ucb.append(self.childNodes[i].wins / self.childNodes[i].visits + 1.0 * sqrt(
                2 * log(self.visits) / self.childNodes[i].visits))
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = pr.choice(indices)
        s = self.childNodes[ind]
        return s

    def Addnode(self, m):
        n = Node(position=m, parent=self)
        self.childNodes.append(n)

    def Update(self, result):
        self.visits += 1
        self.wins += result


def MCTS(root):
    """initialization of the chemical trees and grammar trees"""
    rootnode = Node()
    maxnum = 0
    """----------------------------------------------------------------------"""

    """global variables used for save valid compounds and simulated compounds"""
    valid_compound = []
    max_score = -100.0
    current_score = []
    depth = []

    """----------------------------------------------------------------------"""

    while maxnum < 10:
        node = rootnode
        state = root.Clone()
        """selection step"""
        node_pool = []

        while node.childNodes != []:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print("state position: ", state.position)
        if len(state.position) > 3:
            a = 1
        depth.append(len(state.position))
        if len(state.position) >= 81:
            re = -1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            """------------------------------------------------------------------"""

            """expansion step"""
            """calculate how many nodes will be added under current leaf"""
            expanded = expanded_node(model, state.position, val)  # find the possible element to add
            try:
                expanded.remove(0)
            except:
                None

            nodeadded = node_to_add(expanded, val)  # add the possible element

            all_posible = chem_kn_simulation(model, state.position, val, nodeadded)  # get the branch

            generate_smile = predict_smile(all_posible, val)
            new_compound = make_input_smile(generate_smile)

            node_index, score, valid_smile = check_node_type(new_compound, SA_mean, SA_std, logP_mean,
                                                             logP_std, cycle_mean, cycle_std)

            for s in valid_smile:
                if s not in valid_compound:
                    valid_compound.append(s)
                    print('!!!found ', s)
                    # if node.depth == 2:
                    #     a = 1
            if len(node_index) == 0:
                re = -1.0
                while node != None:
                    node.Update(re)
                    node = node.parentNode
            else:
                re = []
                for i in range(len(node_index)):
                    m = node_index[i]
                    print('runtime: ', maxnum)
                    print("current found max_score:", max_score)
                    print('===========================')
                    print('number of all poossible:', len(all_posible))
                    print('number of valid: ', len(valid_compound))

                    maxnum = maxnum + 1
                    node.Addnode(nodeadded[m])
                    node_pool.append(node.childNodes[i])
                    if score[i] >= max_score:
                        max_score = score[i]
                        current_score.append(max_score)
                    else:
                        current_score.append(max_score)
                    depth.append(len(state.position))
                    """simulation"""
                    re.append((0.8 * score[i]) / (1.0 + abs(0.8 * score[i])))
                    """backpropation step"""
                # print("node pool length:",len(node.childNodes))

                for i in range(len(node_pool)):
                    node = node_pool[i]
                    while node != None:
                        node.Update(re[i])
                        node = node.parentNode

        """check if found the desired compound"""

    print("logp max found:", current_score)
    # print("length of score:",len(current_score))

    print("valid_com=", valid_compound)
    print("num_valid:", len(valid_compound))
    print("depth=", depth)
    print(len(depth))
    return valid_compound


def UCTchemical():
    state = chemical()
    state.Carbon()
    best = MCTS(root=state)

    return best


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "" # use only CPU
    smile_old = zinc_data_with_bracket_original()  # read 250k data as string into the list smile old
    val, smile = zinc_processed_with_bracket(smile_old)

    print(val)
    # val = ['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']

    logP_values = np.loadtxt('logP_values.txt')
    SA_scores = np.loadtxt('SA_scores.txt')
    cycle_scores = np.loadtxt('cycle_scores.txt')
    SA_mean = np.mean(SA_scores)
    SA_std = np.std(SA_scores)
    logP_mean = np.mean(logP_values)
    logP_std = np.std(logP_values)
    cycle_mean = np.mean(cycle_scores)
    cycle_std = np.std(cycle_scores)

    model = loaded_model()
    valid_compound = UCTchemical()
