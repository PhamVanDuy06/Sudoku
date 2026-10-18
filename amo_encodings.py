import math

from pysat.card import CardEnc, EncType

COMMANDER_GROUP_SIZE = 3


def _pairwise_clauses(lits):
    return [[-a, -b] for i, a in enumerate(lits) for b in lits[i + 1:]]


def _pairwise(lits, top_id):
    return _pairwise_clauses(lits), top_id


def _pysat_amo(lits, top_id, encoding):
    cnf = CardEnc.atmost(lits=lits, bound=1, top_id=top_id, encoding=encoding)
    clauses = [list(cl) for cl in cnf.clauses]
    new_top = max([top_id] + [abs(lit) for cl in clauses for lit in cl])
    return clauses, new_top


def _seqcounter(lits, top_id):
    return _pysat_amo(lits, top_id, EncType.seqcounter)


def _bitwise(lits, top_id):
    return _pysat_amo(lits, top_id, EncType.bitwise)


def _commander(lits, top_id):
    clauses, commanders = [], []
    g = COMMANDER_GROUP_SIZE
    for start in range(0, len(lits), g):
        group = lits[start:start + g]
        top_id += 1
        c = top_id
        commanders.append(c)
        clauses += _pairwise_clauses(group)
        clauses += [[-x, c] for x in group]
        clauses.append([-c] + group)
    clauses += _pairwise_clauses(commanders)       # AMO giữa các biến chỉ huy
    return clauses, top_id


def _product(lits, top_id):
    k = len(lits)
    p = math.ceil(math.sqrt(k))
    q = math.ceil(k / p)
    rows = [top_id + i + 1 for i in range(p)]
    cols = [top_id + p + j + 1 for j in range(q)]
    top_id += p + q

    clauses = []
    for idx, x in enumerate(lits):          # biến thứ idx nằm ở ô (idx // q, idx % q)
        clauses.append([-x, rows[idx // q]])
        clauses.append([-x, cols[idx % q]])
    clauses += _pairwise_clauses(rows)
    clauses += _pairwise_clauses(cols)
    return clauses, top_id


ENCODERS = {
    "pairwise": _pairwise,
    "seqcounter": _seqcounter,
    "bitwise": _bitwise,
    "commander": _commander,
    "product": _product,
}
ENCODING_NAMES = list(ENCODERS)


def encode_amo(lits, enc_type, top_id):
    if enc_type not in ENCODERS:
        raise ValueError(f"Không hỗ trợ kiểu encoding: {enc_type}")
    if len(lits) <= 1:
        return [], top_id
    return ENCODERS[enc_type](lits, top_id)