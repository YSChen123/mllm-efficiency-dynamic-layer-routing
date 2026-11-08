import os
import sys
import subprocess
import json
from pathlib import Path
from par_repro.config import resolve


def require_idle_gpu():
    physical = os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    if not physical.isdecimal():
        raise ValueError('Expose one GPU per CLI worker with CUDA_VISIBLE_DEVICES=<ID>')
    value = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    used, utilization = map(int, value.strip().split(','))
    if used >= 1000 or utilization >= 5:
        raise RuntimeError(f'GPU{physical} busy ({used}MiB, {utilization}%); choose an idle GPU')


def load_frozen(workspace, config):
    import torch
    import transformers
    if not torch.__version__.startswith('2.7.1') or transformers.__version__ != '4.37.2':
        raise ValueError('Reference runtime requires Torch2.7.1 and Transformers4.37.2')
    prepared = json.loads((Path(workspace) / 'prepared.json').read_text())
    if prepared['status'] != 'PREPARED' or prepared['model_paths'] != config['model']:
        raise ValueError('Run prepare for these exact model paths before inference')
    for name in ['backbone', 'vision']:
        for filename, expected in prepared['models'][name]['stats'].items():
            actual = (resolve(workspace, config['model'][name]) / filename).stat()
            if (actual.st_size, actual.st_mtime_ns) != (expected['size'], expected['mtime_ns']):
                raise ValueError('Prepared model changed; re-run checksum preparation')
    source = resolve(workspace, config['model']['llava_source'])
    if not (source / 'llava/model/language_model/llava_llama.py').is_file():
        raise FileNotFoundError('Run prepare to obtain pinned LLaVA source: ' + str(source))
    sys.path.insert(0, str(source))
    commit = subprocess.check_output(['git', '-C', str(source.parent), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source.parent), 'status', '--porcelain', '--', 'LLaVA'], text=True).strip()
    if commit != prepared['source_commit'] or dirty:
        raise ValueError('Pinned LLaVA source changed')
    os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init
    import llava
    if not Path(llava.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('Imported LLaVA from a different environment/path')
    disable_torch_init()
    backbone = resolve(workspace, config['model']['backbone'])
    vision_path = resolve(workspace, config['model']['vision'])
    cfg = LlavaConfig.from_pretrained(backbone, local_files_only=True)
    cfg.mm_vision_tower = str(vision_path)
    cfg.image_aspect_ratio = 'pad'
    backend = config['model']['attention_backend']
    cfg._attn_implementation = backend
    model, info = LlavaLlamaForCausalLM.from_pretrained(backbone, config=cfg, torch_dtype=torch.float16,
        device_map={'': 'cuda:0'}, attn_implementation=backend, local_files_only=True, output_loading_info=True)
    if info['missing_keys'] or info['unexpected_keys']:
        raise ValueError('Backbone loading mismatch')
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=False, local_files_only=True)
    vision = model.get_vision_tower()
    vision.load_model(device_map={'': 'cuda:0'})
    vision.to(device='cuda:0', dtype=torch.float16)
    model.eval().requires_grad_(False)
    actual = model.model.layers[0].self_attn.__class__.__name__
    expected = {'sdpa': 'LlamaSdpaAttention', 'eager': 'LlamaAttention'}[backend]
    if actual != expected or len(model.model.layers) != 32 or vision.num_patches != 576:
        raise ValueError('Actual backbone/backend differs from effective config')
    if cfg.mm_use_im_start_end:
        raise ValueError('Reference prompt assumes no image start/end special tokens')
    return model, tokenizer, vision


def make_prompt(tokenizer, question):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    start = (ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
    if start.numel() != 1:
        raise ValueError('Exactly one image required')
    return prompt, ids, int(start.item())


def prepare_image(model, vision, tokenizer, image, question):
    import torch
    from llava.mm_utils import process_images
    prompt, ids, start = make_prompt(tokenizer, question)
    pixels = process_images([image], vision.image_processor, model.config).cuda().half()
    with torch.inference_mode():
        prefix = model.prepare_inputs_labels_for_multimodal(ids, None, torch.ones_like(ids, dtype=torch.bool),
            None, None, pixels, [image.size])[4]
    visual = prefix[:, start:start + 576]
    text = torch.cat([prefix[:, :start], prefix[:, start + 576:]], dim=1)
    return {'prompt': prompt, 'ids': ids, 'pixels': pixels, 'prefix': prefix, 'visual': visual, 'text': text, 'image_start': start}
