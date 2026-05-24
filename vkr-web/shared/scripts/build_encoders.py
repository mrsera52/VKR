from __future__ import annotations

import csv
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path('/data')
OUT_PATH = DATA_DIR / 'encoders.pkl'
F1_THRESHOLD = 0.8
BBOX_DIM = 12


def _find(pattern: str) -> Path:
    for f in DATA_DIR.iterdir():
        if f.is_file() and re.search(pattern, f.name, re.IGNORECASE):
            return f
    raise FileNotFoundError(f'Не найден файл по шаблону "{pattern}" в {DATA_DIR}')


def parse_report(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    header_page_raw, header_q_raw, header_opts = rows[0], rows[1], rows[2]

    header_page, header_q = [], []
    cur_page, cur_q = '', ''
    for p, q in zip(header_page_raw, header_q_raw):
        if p.strip():
            cur_page = p.strip()
        if q.strip():
            cur_q = q.strip()
        header_page.append(cur_page)
        header_q.append(cur_q)

    data_rows = rows[4:]
    col_names = []
    for i, (p, q, o) in enumerate(zip(header_page, header_q, header_opts)):
        parts = [x.strip() for x in [p, q, o] if x.strip()]
        col_names.append(' | '.join(parts) if parts else f'col_{i}')

    name_cnt = Counter(col_names)
    dups = {k: v for k, v in name_cnt.items() if v > 1}
    if dups:
        print(f'  [parse_report] uniquify {len(dups)} dup-names')
        for i, n in enumerate(col_names):
            if name_cnt[n] > 1:
                col_names[i] = f'{n} [col_{i}]'

    records = []
    for row in data_rows:
        if not row or not row[0].strip():
            continue
        rec = {col_names[i]: val.strip() for i, val in enumerate(row) if i < len(col_names)}
        records.append(rec)
    return col_names, records, rows


def parse_criteria(path: Path):
    crit = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['column'] != 'mean':
                crit[row['column']] = {
                    'f1_macro': float(row['f1_macro']),
                    'f1_micro': float(row['f1_micro']),
                    'f1_weighted': float(row['f1_weighted']),
                }
    return crit


def clean_q(q: str) -> str:
    q = re.sub(r'^\d+[\.\s]*', '', q)
    q = re.sub(r'\s*\([^)]*(?:выбор|ответ)[^)]*\)\s*$', '', q, flags=re.IGNORECASE)
    q = q.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', q).strip().lower()


def parse_suffix(name: str):
    m = re.search(r'\.(\d+)$', name)
    return (name[:m.start()], int(m.group(1))) if m else (name, 0)


def normalize(s: str) -> str:
    return s.strip().replace(' ', '').lower()


def sanitize_column(col: str) -> str:
    return re.sub(r'\W|^(?=\d)', '_', col)


def build_criteria_configs(col_names, criteria, target_set, raw_rows):
    row1, row2 = raw_rows[1], raw_rows[2]
    cur_q, cur_type = '', ''
    col_struct = []
    for i in range(len(row1)):
        r1, r2 = row1[i].strip(), row2[i].strip()
        if r1:
            cur_q = r1
            m = re.search(r'\(([^)]*)\)', r1)
            cur_type = m.group(1).lower() if m else ''
        col_struct.append({
            'idx': i, 'q_clean': clean_q(cur_q),
            'option': r2, 'opt_lower': r2.strip().lower(),
            'is_multi': 'множественный' in cur_type,
        })

    criteria_col_map, unmapped = {}, []
    for crit_name in criteria.keys():
        base_name, suffix_num = parse_suffix(crit_name)
        parts = base_name.split(' | ')
        if len(parts) == 2:
            feat, var = parts[0].strip().lower(), parts[1].strip().lower()
            matches = [cs['idx'] for cs in col_struct if cs['opt_lower'] == var and feat in cs['q_clean']]
            if not matches:
                ws = [w for w in feat.split() if len(w) > 2]
                matches = [cs['idx'] for cs in col_struct if cs['opt_lower'] == var
                           and ws and all(w in cs['q_clean'] for w in ws[:2])]
            if matches:
                criteria_col_map[crit_name] = ('binary', matches[suffix_num] if suffix_num < len(matches) else matches[0])
            else:
                unmapped.append(crit_name)
        else:
            feat = parts[0].strip().lower()
            occ = [cs['idx'] for cs in col_struct if not cs['is_multi'] and not cs['option'] and feat == cs['q_clean']]
            if not occ:
                occ = [cs['idx'] for cs in col_struct if not cs['is_multi'] and not cs['option'] and feat in cs['q_clean']]
            if occ:
                criteria_col_map[crit_name] = ('multiclass', occ[suffix_num] if suffix_num < len(occ) else occ[-1])
            else:
                unmapped.append(crit_name)

    configs = []
    for name, m in criteria.items():
        if name not in criteria_col_map:
            continue
        task_type, col_idx = criteria_col_map[name]
        configs.append({
            'name': name,
            'task_info': {'type': task_type, 'col_idx': col_idx},
            'priority': 'high' if name in target_set else 'normal',
        })
    configs.sort(key=lambda x: x['name'])
    return configs, unmapped


def majority_label(values, task_type):
    if task_type == 'binary':
        pos = sum(1 for v in values if v)
        return 1 if pos * 2 >= len(values) and pos > 0 else 0
    non_empty = [v for v in values if str(v).strip()]
    if not non_empty:
        return ''
    ctr = Counter(non_empty)
    top_cnt = ctr.most_common(1)[0][1]
    top_vals = [v for v, c in ctr.items() if c == top_cnt]
    if len(top_vals) == 1:
        return top_vals[0]
    for v in non_empty:
        if v in top_vals:
            return v
    return non_empty[0]


def main():
    print(f'data dir: {DATA_DIR}')

    report_path = _find(r'^report.*\.csv$')
    results_path = _find(r'^results.*\.csv$')
    print(f'report:  {report_path.name}')
    print(f'results: {results_path.name}')

    col_names, records, raw_rows = parse_report(report_path)
    criteria = parse_criteria(results_path)
    code_col = col_names[9]

    target_set = {n for n, m in criteria.items() if m['f1_macro'] < F1_THRESHOLD}
    print(f'records: {len(records)}, criteria: {len(criteria)}, target (F1<{F1_THRESHOLD}): {len(target_set)}')

    criteria_configs, unmapped = build_criteria_configs(col_names, criteria, target_set, raw_rows)
    print(f'mapped: {len(criteria_configs)}/{len(criteria)} (unmapped: {len(unmapped)})')

    records_by_code = defaultdict(list)
    for rec in records:
        code = rec.get(code_col, '').strip()
        if not code:
            continue
        records_by_code[code].append(rec)

    dataset_for_enc = []
    for code, recs in records_by_code.items():
        labels = {}
        for cfg in criteria_configs:
            n = cfg['name']
            ci = cfg['task_info']['col_idx']
            col = col_names[ci]
            vals = [r.get(col, '').strip() for r in recs]
            labels[n] = majority_label(vals, cfg['task_info']['type'])
        dataset_for_enc.append({'code': code, 'labels': labels})

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        binary_names = [c['name'] for c in criteria_configs if c['task_info']['type'] == 'binary']
        multi_names = [c['name'] for c in criteria_configs if c['task_info']['type'] == 'multiclass']
        target_names_set = {c['name'] for c in criteria_configs if c['priority'] == 'high'}
        N = len(dataset_for_enc)
        rare_cols = []
        for bn in binary_names:
            if bn not in target_names_set:
                continue
            col = [1 if dataset_for_enc[i]['labels'].get(bn) == 1 else 0 for i in range(N)]
            s = sum(col)
            if 1 < s < N:
                rare_cols.append(col)
        for mn in multi_names:
            if mn not in target_names_set:
                continue
            ctr = Counter(dataset_for_enc[i]['labels'].get(mn, '') for i in range(N))
            for cls, cnt in ctr.items():
                if cls and 1 < cnt < N and cnt / N < 0.3:
                    rare_cols.append([1 if dataset_for_enc[i]['labels'].get(mn) == cls else 0 for i in range(N)])

        if rare_cols:
            Y_e = np.array(rare_cols, dtype=int).T
            X_e = np.arange(N).reshape(-1, 1)
            mss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
            train_idx_enc, _ = next(mss1.split(X_e, Y_e))
        else:
            rng = np.random.default_rng(42)
            perm = rng.permutation(N)
            train_idx_enc = perm[:int(N * 0.70)]
    except ImportError:
        print('iterstrat не установлен; используется простой случайный сплит')
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(dataset_for_enc))
        train_idx_enc = perm[:int(len(dataset_for_enc) * 0.70)]

    train_data_enc = [dataset_for_enc[i] for i in train_idx_enc]

    label_encoders = {}
    for cfg in criteria_configs:
        if cfg['task_info']['type'] != 'multiclass':
            continue
        name = cfg['name']
        vals = sorted({str(it['labels'].get(name, '')).strip() for it in train_data_enc
                       if str(it['labels'].get(name, '')).strip()})
        enc = {'': 0}
        for i, v in enumerate(vals, 1):
            enc[v] = i
        label_encoders[name] = enc

    label_decoders = {n: {i: lbl for lbl, i in enc.items()} for n, enc in label_encoders.items()}

    target_cols = [c['name'] for c in criteria_configs]
    safe_col_map = {c: sanitize_column(c) for c in target_cols}
    num_classes_safe = {}
    for cfg in criteria_configs:
        n = cfg['name']
        sn = safe_col_map[n]
        if cfg['task_info']['type'] == 'binary':
            num_classes_safe[sn] = 2
        else:
            num_classes_safe[sn] = max(len(label_encoders.get(n, {})), 2)

    target_safe_names = {safe_col_map[c['name']] for c in criteria_configs if c['priority'] == 'high'}
    nontarget_safe_names = {safe_col_map[c['name']] for c in criteria_configs if c['priority'] != 'high'}

    bbox_mean = np.array([0.78, 0.78, 0.62, 0.50, 0.50, 1.00,
                          0.05, 0.05, 0.05, 0.05, 0.05, 0.90], dtype=np.float32)
    bbox_std = np.array([0.18, 0.18, 0.25, 0.10, 0.10, 0.30,
                         0.22, 0.22, 0.22, 0.22, 0.10, 0.30], dtype=np.float32)

    payload = {
        'criteria_configs': criteria_configs,
        'label_encoders': label_encoders,
        'label_decoders': label_decoders,
        'safe_col_map': safe_col_map,
        'num_classes_safe': num_classes_safe,
        'target_safe_names': target_safe_names,
        'nontarget_safe_names': nontarget_safe_names,
        'bbox_mean': bbox_mean,
        'bbox_std': bbox_std,
        'bbox_dim': BBOX_DIM,
    }

    OUT_PATH.write_bytes(pickle.dumps(payload))
    print(f'\nencoders.pkl сохранён: {OUT_PATH}')
    print(f'  размер:        {OUT_PATH.stat().st_size / 1024:.1f} КБ')
    print(f'  criteria:      {len(criteria_configs)}')
    print(f'  multiclass:    {len(label_encoders)}')
    print(f'  high-priority: {len(target_safe_names)}')


if __name__ == '__main__':
    main()
