"""Adapter-independent rank-one edits using each base matrix's 50% energy space."""
import re

from pilot_eval.stage0_edit_math import edit_module
from pilot_eval.stage0_spectral_math import decompose

OPERATIONS = {
    'output': 'V1', 'input': 'V2',
    'random0': 'V3-0', 'random1': 'V3-1', 'random2': 'V3-2',
    'output-magnitude': 'V4', 'input-magnitude': 'V4-input',
}


def variant_catalogue(arms):
    arms = list(arms)
    if not arms or len(set(arms)) != len(arms) or any(
        not isinstance(x, str) or not re.fullmatch(r'[a-z0-9_]+', x) for x in arms
    ):
        raise ValueError('Arm names must be unique, nonempty safe identifiers.')
    return {f'{arm}-{op}': {'source_arm': arm, 'operation': op}
            for arm in arms for op in ('original', *OPERATIONS)}


def module_edits(matrix, adapters):
    """Edit one module across arms, computing principal bases once per base matrix."""
    variant_catalogue(adapters)
    arrays, spectrum = decompose(matrix)
    k = spectrum['k50']
    u, v = arrays['u'][:, :k], arrays['v'][:, :k]
    edits, checks = {}, {}
    for arm, factors in adapters.items():
        results, evidence, _ = edit_module(*factors, u, v)
        for operation, key in OPERATIONS.items():
            edits[f'{arm}-{operation}'] = results[key]
            checks[f'{arm}-{operation}'] = evidence[key]
    return edits, {'spectrum': spectrum, 'checks': checks}
