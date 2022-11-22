"""
Frequently used utility functions
"""


def counts2probs(counts):
    """Convert a dictionary of experimental counts or a list of
    dictionaries to probabilities.

    Parameters
    ----------
    counts : Counts, list[Counts] or dict, list[dict]
        Experimental counts.

    Returns
    -------
    dict or list[dict]
        Experimental probabilities.
    """
    
    probs   = None
    is_list = (type(counts)==list)
    
    if is_list:
        num   = len(counts)
        probs = [None]*num
        for i in range(num):
            counts_  = counts[i]
            shots    = sum(counts_.values())
            probs_   = {b: c/shots for b, c in counts_.items()}
            probs[i] = probs_
    else:
        shots = sum(counts.values())
        probs = {b: c/shots for b, c in counts.items()}
        
    
    return probs