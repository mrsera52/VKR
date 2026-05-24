from collections import Counter, defaultdict

def compute_trait_support(predictions, criteria_dict):
    support = Counter()
    evidence = defaultdict(list)
    for crit_name, value in predictions.items():
        if value in (0, '', None):
            continue
        entry = criteria_dict['criteria'].get(crit_name)
        if not entry:
            continue
        if entry['kind'] == 'binary_presence':
            traits = entry.get('traits', [])
        else:
            traits = entry.get('class_traits', {}).get(value, [])
        for t in traits:
            support[t] += 1
            evidence[t].append((crit_name, value))
    return support, evidence
