from fractions import Fraction
from collections import Counter
from itertools import permutations
from typing import List, Set
import random


# ---- EZpoints mapping ----
EZ_TOKENS = ['1','2','3','4','5','6','7','8','9','10','+','*','=']
EZ_TOKEN_TO_ACTION = {t:i+1 for i,t in enumerate(EZ_TOKENS)}
EZ_NUMS = set(EZ_TOKENS[:10])
TARGET12 = Fraction(12, 1)

def _clean(tokens: List[str|int]) -> List[str]:
    return [str(t).strip() for t in tokens if str(t).strip()]

def _num_counter(nums: List[int]) -> Counter:
    return Counter(int(n) for n in nums)

def _used_nums(tokens: List[str], num_set: Set[str]) -> Counter:
    return Counter(int(t) for t in tokens if t in num_set)

def _uses_exact(tokens: List[str], numbers: List[int], num_set: Set[str]) -> bool:
    return _used_nums(tokens, num_set) == _num_counter(numbers)

def _eval_tokens_ez(tokens: List[str]):
    # Only numbers, +, * are expected; equals is not part of the expression
    expr = []
    for t in tokens:
        if t in EZ_NUMS:
            expr.append(f'Fraction({int(t)},1)')
        elif t in {'+','*'}:
            expr.append(t)
        elif t == '=':
            break
        else:
            return None
    if not expr:
        return None
    try:
        return eval(''.join(expr), {'Fraction': Fraction})
    except Exception:
        return None

def _solutions_for_ez(nums: List[int]) -> List[List[str]]:
    """All token sequences using each number exactly once that evaluate to 12."""
    sols = []
    for a,b in set(permutations(nums, 2)):
        if a + b == 12:
            sols.append([str(a), '+', str(b)])
        if a * b == 12:
            sols.append([str(a), '*', str(b)])
    # Deduplicate & stable
    uniq, seen = [], set()
    for s in sols:
        t = tuple(s)
        if t not in seen:
            seen.add(t); uniq.append(s)
    return uniq

def _random_action_ez(numbers: List[int], prefix: List[str]) -> int:
    # Don’t overuse numbers; allow +, * freely; '=' only when already valid and complete
    need = _num_counter(numbers)
    used = _used_nums(prefix, EZ_NUMS)
    avail_nums = [str(n) for n,c in need.items() for _ in range(max(0, c - used[n]))]
    choices = avail_nums + ['+','*']
    val_now = _eval_tokens_ez(prefix)
    if val_now == TARGET12 and _uses_exact(prefix, numbers, EZ_NUMS):
        choices.append('=')
    tok = random.choice(choices if choices else EZ_TOKENS[:-1])  # avoid '=' if nothing valid
    return EZ_TOKEN_TO_ACTION[tok]

def get_next_action_for_ezpoints(numbers: List[int], formula: List[str|int]) -> int:
    """
    Oracle for EZpoints (2 numbers, + and *, target=12).
    Returns the next action index from the EZ action set.
    If no matching winning plan exists for the current prefix, picks a random sensible action.
    """
    prefix = _clean(formula)

    # If already complete & correct -> '='
    if _eval_tokens_ez(prefix) == TARGET12 and _uses_exact(prefix, numbers, EZ_NUMS):
        return EZ_TOKEN_TO_ACTION['=']

    # Continue a consistent winning plan if possible
    for sol in _solutions_for_ez(numbers):
        if prefix == sol[:len(prefix)]:
            return EZ_TOKEN_TO_ACTION[sol[len(prefix)]]

    # Otherwise random fallback
    return _random_action_ez(numbers, prefix)
