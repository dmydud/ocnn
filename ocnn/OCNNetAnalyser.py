import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import hmean
import scipy


class OCNNetAnalyser:
    def __init__(self):
        pass

    @staticmethod
    def bit_mutual_info(seq1: np.ndarray, seqs2: np.ndarray) -> np.ndarray:
        if seq1.ndim != 1 or seqs2.ndim != 2:
            raise ValueError

        length = seq1.size

        tt = np.logical_and(seq1, seqs2).sum(axis=1)
        ff = length - np.logical_or(seq1, seqs2).sum(axis=1)
        ft = seqs2.sum(axis=1) - tt
        tf = np.logical_xor(seq1, seqs2).sum(axis=1) - ft

        dependent_props = np.array([[ff, ft], [tf, tt]]) / length

        independent_props = (dependent_props.sum(axis=1).reshape(2, 1, -1)
                             * dependent_props.sum(axis=0).reshape(1, 2, -1))

        zeros = np.zeros_like(dependent_props)
        dep_indep_props_ratio = np.divide(dependent_props, independent_props,
                                          where=independent_props != 0, out=zeros)
        log_val = np.log(dep_indep_props_ratio, where=dep_indep_props_ratio != 0, out=zeros)
        return np.sum(dependent_props * log_val, axis=0).sum(axis=0)

    @staticmethod
    def calc_mutual_info(net_states):
        net_size = net_states.shape[0]
        mutual_info = np.zeros((net_size, net_size), dtype=np.float32)

        net_bin_states = net_states > 0

        for osc_i in range(net_size):
            curr_osc = net_bin_states[osc_i]
            next_oscs = net_bin_states[osc_i + 1:]
            mutual_info[osc_i, osc_i + 1:] = OCNNetAnalyser.bit_mutual_info(curr_osc, next_oscs)

        return mutual_info + mutual_info.T

    @staticmethod
    def get_linkage(net_states):
        mutual_info = OCNNetAnalyser.calc_mutual_info(net_states)

        tmp = np.log(2) - mutual_info
        np.fill_diagonal(tmp, val=0)

        return linkage(scipy.spatial.distance.squareform(tmp))

    @staticmethod
    def cluster_by_linkage(linkage_matrix, theta):
        return fcluster(linkage_matrix, np.log(2) - theta, criterion="distance")

    @staticmethod
    def cluster_by_mutual_info(mutual_info, theta):
        graph = csr_matrix(np.tril(mutual_info, 1) > theta)
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        return labels

    @staticmethod
    def cluster(net_states, theta):
        mutual_info = OCNNetAnalyser.calc_mutual_info(net_states)
        return OCNNetAnalyser.cluster_by_mutual_info(mutual_info, theta)
