"""Add our external aliases, then execute the pinned author's evaluation CLI."""
import lmms_eval.models
lmms_eval.models.AVAILABLE_MODELS['llava_roe_dynamic']='roe_repro.dynamic_model.DynamicLlava'
lmms_eval.models.AVAILABLE_MODELS['llava_roe_protected']='roe_repro.dynamic_model_protected.DynamicProtectedLlava'
from lmms_eval.__main__ import cli_evaluate

if __name__=='__main__':cli_evaluate()
