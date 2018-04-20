from math import *
import numpy as np
import random as pr
import time
from load_model import loaded_model
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile, predict_smile, check_node_type, node_to_add, \
    expanded_node
import os

class chemical:

    def __init__(self):
        self.position = ['&']

    def Clone(self):
        st = chemical()
        st.position = self.position[:]
        return st

    def SelectPosition(self, m):
        self.position.append(m)


class Node:

    def __init__(self, position=None, parent=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child = None
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
    start_time = time.time()
    """----------------------------------------------------------------------"""

    """global variables used for save valid compounds and simulated compounds"""
    valid_compound = []
    all_simulated_compound = []
    max_score = -100.0
    current_score = []
    depth = []
    all_score = []

    """----------------------------------------------------------------------"""

    while maxnum < 10100:
        print('runtime: ', maxnum)
        node = rootnode
        state = root.Clone()
        """selection step"""
        node_pool = []
        print("current found max_score:", max_score)

        while node.childNodes != []:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        print("state position: ", state.position)
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
            expanded = expanded_node(model, state.position, val)
            nodeadded = node_to_add(expanded, val)

            all_posible = chem_kn_simulation(model, state.position, val, nodeadded)
            generate_smile = predict_smile(all_posible, val)
            new_compound = make_input_smile(generate_smile)

            node_index, score, valid_smile, all_smile = check_node_type(new_compound, SA_mean, SA_std, logP_mean,
                                                                        logP_std, cycle_mean, cycle_std)

            print('node index: ', node_index)
            valid_compound.extend(valid_smile)
            all_simulated_compound.extend(all_smile)
            all_score.extend(score)
            if len(node_index) == 0:
                re = -1.0
                while node != None:
                    node.Update(re)
                    node = node.parentNode
            else:
                re = []
                for i in range(len(node_index)):
                    m = node_index[i]
                    print('===========================')
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
                    if maxnum == 100:
                        maxscore100 = max_score
                        time100 = time.time() - start_time
                    if maxnum == 500:
                        maxscore500 = max_score
                        time500 = time.time() - start_time
                    if maxnum == 1000:
                        maxscore1000 = max_score
                        time1000 = time.time() - start_time
                    if maxnum == 5000:
                        maxscore5000 = max_score
                        time5000 = time.time() - start_time
                    if maxnum == 10000:
                        time10000 = time.time() - start_time
                        maxscore10000 = max_score
                    """backpropation step"""
                # print("node pool length:",len(node.childNodes))

                for i in range(len(node_pool)):
                    node = node_pool[i]
                    while node != None:
                        node.Update(re[i])
                        node = node.parentNode

            # print("four step time:",finish_iteration_time)

        """check if found the desired compound"""

    # print("all valid compounds:",valid_compound)

    finished_run_time = time.time() - start_time

    print("logp max found:", current_score)
    # print("length of score:",len(current_score))
    # print("time:",time_distribution)

    print("valid_com=", valid_compound)
    print("num_valid:", len(valid_compound))
    print("all compounds:", len(all_simulated_compound))
    print("score=", all_score)
    print("depth=", depth)
    print(len(depth))
    print("runtime", finished_run_time)
    # print("num_searched=",num_searched)
    print("100 max:", maxscore100, time100)
    print("500 max:", maxscore500, time500)
    print("1000 max:", maxscore1000, time1000)
    print("5000 max:", maxscore5000, time5000)
    print("10000 max:", maxscore10000, time10000)
    return valid_compound


def UCTchemical():
    state = chemical()
    best = MCTS(root=state)

    return best


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    smile_old = zinc_data_with_bracket_original()  # read 250k data as string into the list smile old
    val, smile = zinc_processed_with_bracket(smile_old)
    print(val)

    logP_values = np.loadtxt('logP_values.txt')
    SA_scores = np.loadtxt('SA_scores.txt')
    cycle_scores = np.loadtxt('cycle_scores.txt')
    SA_mean = np.mean(SA_scores)
    print(len(SA_scores))

    SA_std = np.std(SA_scores)
    logP_mean = np.mean(logP_values)
    logP_std = np.std(logP_values)
    cycle_mean = np.mean(cycle_scores)
    cycle_std = np.std(cycle_scores)

    model = loaded_model()
    valid_compound = UCTchemical()
