import numpy as np
from Operator import *

def context_operator_function(e, i, f_k) :
    """
    :param e:    Operator                                            -- Contextual Edit Operator
    :param i:    Integer                                             -- Index in S
    :param f_k:  (int k * int x * float sigma  -> Float in [0,1])    -- Context Function

    :return:     Float                                               -- Value of \phi_e(x) in [0,1]
    """
    if(e.op == Edit_operator.MOD) :
        return f_k(e.k, i, len(e.S))
    elif(e.op == Edit_operator.ADD) :
        if(i <= e.k - 1) :
            return f_k(e.k, i + 1, len(e.S))
        else :
            return f_k(e.k, i, len(e.S))
    else : # Case e == DEL
            if (i <= e.k - 1):
                return f_k(e.k, i + 1, len(e.S))
            elif (i == e.k and len(e.S) != 1):
                return 0
            else :
                return f_k(e.k, i - 1, len(e.S))

def edit_cost(e, f_k, sim, alpha):
    """
    :param e: Operator(Edit_operator * List<T> * T * Int) -- Contextual Edit Operation
    :return:  Float                                       --  \gamma(e) in [0,1]
    """
    sim_v = list(map(lambda s: sim(s, e.a), e.S)) # Similarity vector
    ctx_v = list(map(lambda i: context_operator_function(e, i, f_k), range(len(sim_v)))) # Context vector
    for i in range(len(sim_v)):
        sim_v[i] *= ctx_v[i]
    delta_cost = 1 if e.op != Edit_operator.MOD else 1 - sim(e.S[e.k], e.a)
    gamma = (alpha * delta_cost) + (1 - alpha) * (1 - max(sim_v))
    return gamma

def one_sided_CED(S1, S2, sim, f_k, alpha):
    """
    :return: Float -- \tilde{d}_{CED}(S_1,S_2) Cost to edit S1 -> S2
    """
    D = np.zeros((len(S1)+1, len(S2)+1))
    for i in range(len(S1) + 1):
        for j in range(len(S2) +1):
            if(i == 0 or j == 0):
                D[i, j] = j + i
            else:
                op_mod = Operator(Edit_operator.MOD, S1, S2[j-1], i-1)
                op_del = Operator(Edit_operator.DEL, S1, S1[i-1], i-1)
                op_add = Operator(Edit_operator.ADD, S1, S2[j-1], i-1)

                cost_mod = edit_cost(op_mod, f_k, sim, alpha)
                cost_del = edit_cost(op_del, f_k, sim, alpha)
                cost_add = edit_cost(op_add, f_k, sim, alpha)

                D[i, j] = round(min(D[i - 1, j-1] + cost_mod,
                              D[i - 1, j] + cost_del,
                              D[i, j - 1] + cost_add), 2)
    return D[len(S1), len(S2)]

def CED(S1, S2, sim, f_k, alpha = 0):
    """
    :param S1: List<T> -- Sequence S1
    :param S2: List<T> -- Sequence S2
    :param sim: (T * T) -> Float -- Similarity measure
    :param f_k: (Int * Float * Float)  -> Float -- Context Function
    :param alpha: Float

    :return: Numpy 2D array Float -- Cost to edit S1 -> S2
    """
    return max(one_sided_CED(S1, S2, sim, f_k, alpha), one_sided_CED(S2, S1, sim, f_k, alpha))

