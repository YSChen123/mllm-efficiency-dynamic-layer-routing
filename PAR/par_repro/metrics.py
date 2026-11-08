def pope_answer(text):
    text = text.split('.')[0].replace(',', '')
    words = text.split(' ')
    return 'no' if 'No' in words or 'not' in words or 'no' in words else 'yes'


def pope_scores(rows):
    if set(row['category'] for row in rows) != {'adversarial', 'popular', 'random'}:
        raise ValueError('All three POPE categories required')
    groups = {}
    for category in ['adversarial', 'popular', 'random']:
        part = [r for r in rows if r['category'] == category]
        tp = fp = tn = fn = 0
        for row in part:
            predicted = pope_answer(row['text']) == 'yes'
            actual = row['label'] == 'yes'
            tp += int(predicted and actual)
            fp += int(predicted and not actual)
            tn += int(not predicted and not actual)
            fn += int(not predicted and actual)
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        groups[category] = {'count': len(part), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'precision': precision,
                            'recall': recall, 'F1': f1, 'accuracy': (tp + tn) / len(part), 'yes_ratio': (tp + fp) / len(part)}
    return {'categories': groups, 'macro_f1_percent': 100 * sum(v['F1'] for v in groups.values()) / 3,
            'macro_accuracy_percent': 100 * sum(v['accuracy'] for v in groups.values()) / 3}


def seed_scores(rows):
    if not rows:
        raise ValueError('Empty SEED predictions')
    correct = sum(r['text'] == r['answer'] for r in rows)
    groups = {}
    for row in rows:
        group = groups.setdefault(str(row['category']), {'correct': 0, 'total': 0})
        group['total'] += 1
        group['correct'] += int(row['text'] == row['answer'])
    for group in groups.values():
        group['accuracy_percent'] = 100 * group['correct'] / group['total']
    return {'correct': correct, 'total': len(rows), 'accuracy_percent': 100 * correct / len(rows), 'by_question_type': groups}
