# Third-party sources and attribution

- PAR: [ASGO-MM-Lab/Pruning-All-Rounder](https://github.com/ASGO-MM-Lab/Pruning-All-Rounder), ICCV2025. Method and public implementation references; source revisions pinned in configs. The downloaded baseline retains its upstream license and notices.
- LLaVA: [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA), used through the unmodified baseline included in PAR commit6d9b39acdce382c4da228345fb6c7dc6736b8899. Model weights retain their own licenses and use restrictions; this repository does not redistribute them.
- [FastV](https://github.com/pkunlp-icler/FastV): reference evaluation prompt/protocol and POPE scoring comparison.
- [POPE](https://github.com/AoiDragon/POPE), [COCO](https://cocodataset.org/), [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench): download original data separately and comply with their licenses. Only ordered sample identifiers and non-reversible image checksums are tracked here; no images or question corpus are redistributed.
- PyTorch, Hugging Face Transformers, CLIP, and dependencies retain their respective licenses.

The numerical reconstruction modules in this branch were created for the reproduction; they are not represented as author-supplied PAR training code. configs/kernel_origins.json records their local provenance and import-only packaging transformations. Mathematical kernels were reused to minimize numerical drift, with independent parity checks documented in docs/REPRODUCIBILITY.md.
