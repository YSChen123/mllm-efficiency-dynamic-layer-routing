"""External image-only task adapter; model and scoring remain official."""
from lmms_eval.tasks.seedbench.utils import seed_process_result, seed_aggregation_result


def select_image_docs(dataset):
    # Read only the type column; do not decode every image while filtering.
    indices = [i for i, kind in enumerate(dataset['data_type']) if kind == 'image']
    return dataset.select(indices)


def image_result(doc, results):
    return {'seed_image': seed_process_result(doc, results)['seed_image']}


image_aggregation = seed_aggregation_result
