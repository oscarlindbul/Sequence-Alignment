import numpy as np
from collections import deque


class Align_Tree:
    max_depth = 0

    def __init__(self, val, root=None):
        self.value = val
        self.children = []
        self.parent = None

        self.depth = 0


        if not root:
            self.root = self
            self.num_nodes = 1
        else:
            self.root = root

    def add_child(self, val):
        self.children.append(Align_Tree(val, root=self.root))

        self.children[-1].depth = self.depth + 1
        if self.children[-1].depth > self.root.max_depth:
            self.root.max_depth = self.children[-1].depth
        if len(self.children) > 1:
            self.root.num_nodes += 1
        return self.children[-1]

    def count_recur(self):
        if len(self.children) == 0:
            return 1
        else:
            count = 1
            for c in self.children:
                count *= c.count_recur()
            return count

    def get_sol_count(self):
        return self.count_recur()

    def align_recur(self, container, current, ind, max_solv):
        current.appendleft(self.value)
        if len(self.children) > 0 and ind[0] < max_solv:
            for i in range(len(self.children)):
                if i > 0:
                    ind[0] += 1
                if ind[0] >= max_solv:
                    break
                self.children[i].align_recur(container, current, ind, max_solv)
        elif len(self.children) == 0:
            container[ind[0]] = current.copy()
        current.popleft()


    def get_aligns(self, max_solv=np.inf):
        container = [deque() for i in range(min(self.num_nodes, max_solv))]
        current = deque()
        for i in range(len(self.children)):
            ind = [0]
            self.children[i].align_recur(container, current, ind, len(container))
            ind[0] += 1
        return container

    def sol_num(self):
        return self.root.num_nodes


def solve_MSA(sequences, costs):
    """Solve the MSA problem using generalized Needleman-Wunsch

    Args:
        sequences - The DNA sequences to align : string list
        costs - the cost of 0:match 1:mismatch, 2:gap : float list

    Returns:
        position list of each sequence element
    """

    dim = len(sequences)

    # allocate position list
    pos_list = []
    sizes = np.zeros(dim, dtype=np.int32)
    for i in range(dim):
        l = len(sequences[i])
        pos_list.append([0]*l)
        sizes[i] = l

    # allocate score matrix
    scores = np.zeros(tuple(sizes+1))
    # cost constants
    match_cost, mismatch_cost, indel_cost = costs[:]

    # helper functions
    def pair_score(c1, c2):
        if c1 == c2 and c2 != "-":
            return match_cost
        elif c1 == "-" and c2 == "-":
            return 0
        elif c1 == "-" or c2 == "-":
            return indel_cost
        else:
            return mismatch_cost

    def smp_score(chars):
        s = 0
        for i in range(len(chars)):
            for j in range(i):
                s += pair_score(chars[i],chars[j])
        return s

    def get_seq_str(pos, gaps):
        nonlocal sequences
        chars = [0]*len(pos)
        for i in range(len(pos)):
            if gaps[i] == 0:
                chars[i] = sequences[i][pos[i]-1]
            else:
                chars[i] = "-"
        return chars

    def to_bin_list(gap_list):
        for i in range(len(gap_list)):
            bin_string = format(gap_list[i], "0" + str(len(sequences)) + "b")
            gap_list[i] = np.array([x == '1' for x in bin_string], dtype=np.bool)

    gap_combos = list(range(2**len(sequences)-1))
    to_bin_list(gap_combos)

    # set score matrix
    def score_recur(pos, dim):
        if dim >= len(pos):
            nonlocal scores
            fixed_gaps = np.where(pos == 0, True, False)
            max_score = -np.inf
            for i in range(len(gap_combos)):
                equals = fixed_gaps == gap_combos[i]
                differ = np.logical_and(fixed_gaps, equals)
                same_fixed = np.logical_or(np.logical_not(fixed_gaps), differ)
                if np.all(same_fixed):
                    gaps = gap_combos[i]
                    column = get_seq_str(pos, gaps)
                    new_score = scores[tuple(pos + gaps - 1)] + smp_score(column)
                    max_score = max(new_score, max_score)
            if max_score == -np.inf:
                max_score = 0
            scores[tuple(pos)] = max_score
            return
        for i in range(sizes[dim]+1):
            pos[dim] = i
            score_recur(pos, dim+1)

    # set score matrix
    pos = np.zeros(len(scores.shape), dtype=np.int32)
    score_recur(pos, 0)

    def trace_recur(pos, align_node):
        if np.all(pos == 0):
            return
        for i in range(len(gap_combos)):
            gaps = gap_combos[i]
            if not np.all(np.logical_or(gaps, pos != 0)):
                continue
            column = get_seq_str(pos, gaps)
            new_pos = pos + gaps - 1
            neigh_score = scores[tuple(new_pos)] + smp_score(column)
            if scores[tuple(pos)] == neigh_score:
                child = align_node.add_child(column)
                trace_recur(new_pos, child)

    def find_trace():
        # find minimas at edge
        pos = np.array(sizes)
        traces = [np.copy(pos)]
        # max_cost = scores[tuple(pos)]
        # max_pos = np.copy(pos)
        # for i in range(len(pos)):
        #     for j in range(sizes[i]):
        #         pos[i] = j
        #         score = scores[tuple(pos)]
        #         if score > max_cost:
        #             copy = np.copy(pos)
        #             traces = [copy]
        #             max_cost = score
        #             max_pos = copy
        #         elif score == max_cost:
        #             traces.append(np.copy(pos))
        #     pos[i] = sizes[i]
        # for i in range(len(traces)):
        #     print("Found max at ", traces[i])
        trace_roots = [Align_Tree(0) for i in range(len(traces))]
        for i in range(len(trace_roots)):
            trace_recur(traces[i], trace_roots[i])
        return trace_roots

    # traceback
    print("Beginning traceback")
    roots = find_trace()

    ## print found path
    print(sum([r.sol_num() for r in roots]), "solutions found")
    for i in range(len(roots)):
        alignments = roots[i].get_aligns()
        align_strings = [[["" for i in range(len(alignments[k]))] for i in range(len(sequences))] for k in range(len(alignments))]
        for j in range(len(alignments)):
            print("alignment", j+1, "/", len(alignments))
            for s in range(len(alignments[j])):
                column = alignments[j].popleft()
                for k in range(len(sequences)):
                    align_strings[j][k][s] = column[k]
            for k in range(len(sequences)):
                align_strings[j][k] = "".join(align_strings[j][k])
                print(align_strings[j][k])
            print()
