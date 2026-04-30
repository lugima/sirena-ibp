def fill_zero_indices(sint):
    """ Fills missing beta entries in a sum-integral with zeroes. """

    if len(sint[0]) > len(sint[1]):
        fill_zeros = len(sint[0]) - len(sint[1])
        beta = tuple( list(sint[1]) + [0] * fill_zeros)
    else:
        beta = sint[1]
    return (sint[0], beta, sint[2])
        

def neighbor_sints(sints):
    """ Yields a list of neighbors for a given input list of sum-integrals.

    A neighbor of a sum-integral is another sum-integral with the same sum of propagator powers,
    but with one propagator raised to one greater power and another raised to one more negative power.
    """

    new_sints = set()
    for sint in sints:
        new_sints.add(sint)
        sum_neg = sum(1 for els in sint[0] if els<0)
        for i in range(len(sint[0])):
            for j in range(len(sint[0])):
                new_sint = list(list(els) for els in sint)
                new_sint[0][i] += 1
                new_sint[0][j] -= 1
                if sum(1 for els in sint[0] if els<0) > sum_neg:
                    pass
                else:
                    new_sints.add(tuple(tuple(els) for els in new_sint))

    return list(new_sints)