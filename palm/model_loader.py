#!/usr/bin/env python3
"""
Unified loader for the heterogeneous PIT model zoo.

Three incompatible APIs:
  1. Standard HF (gpt2, Qwen, Llama, ...)          -> AutoModelForCausalLM
  2. PIT-4B (Diamegs/*)  custom code, auto_map     -> AutoModelForCausalLM(trust_remote_code)
  3. ChronoGPT (manelalab/chrono-gpt-v1-*)         -> PyTorchModelHubMixin, bespoke
     forward(inputs) -> (logits, layer_outputs); no auto_map, no HF config

All three are GPT-2 BPE (verified: PIT merges.txt is MD5-identical to gpt2;
ChronoGPT uses tiktoken.get_encoding("gpt2")), so one tokenizer serves all.

Returns (tokenizer, fwd) where fwd(input_ids: LongTensor[B,T]) -> logits[B,T,V].
"""
import os, sys, types
import torch

# Where transformers keeps its downloads. HF_HOME is what the library itself
# reads, and its own default is ~/.cache/huggingface, so we agree with it
# rather than inventing a second location.
CACHE = os.environ.get("HF_HOME",
                       os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", CACHE)
_CHRONO_SRC = os.path.join(CACHE, "_chrono_src")


def _ensure_chrono_module(trainable=False):
    """
    Fetch ChronoGPT_inference.py once and make it importable.

    trainable=True strips the @torch.inference_mode() decorators, which the
    upstream file puts on every forward. Those decorators make autograd
    impossible, so the contamination experiment (E2) cannot run without this.
    Weights and numerics are untouched; only the no-grad context is removed.
    """
    os.makedirs(_CHRONO_SRC, exist_ok=True)
    name = "ChronoGPT_trainable" if trainable else "ChronoGPT_inference"
    path = os.path.join(_CHRONO_SRC, name + ".py")
    if trainable and not os.path.exists(path):
        base = _ensure_chrono_module(trainable=False)
        src = open(base.__file__).read()
        import re as _re
        src = _re.sub(r"^[ \t]*@torch\.inference_mode\(\)[ \t]*\n", "", src, flags=_re.M)
        open(path, "w").write(src)
    if trainable:
        if _CHRONO_SRC not in sys.path:
            sys.path.insert(0, _CHRONO_SRC)
        import importlib
        return importlib.import_module(name)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        src = hf_hub_download("manelalab/chrono-gpt-v1-20241231",
                              filename="ChronoGPT_inference.py")
        import shutil; shutil.copy(src, path)
    if _CHRONO_SRC not in sys.path:
        sys.path.insert(0, _CHRONO_SRC)
    import ChronoGPT_inference
    return ChronoGPT_inference


def is_chrono(model_id):
    return "chrono-gpt" in model_id


def load(model_id, device="cuda", dtype=torch.bfloat16, trainable=False):
    """`model_id` may carry a git revision as "repo@revision" (Pythia step
    checkpoints are published that way), which is stripped off and passed on."""
    from transformers import AutoTokenizer

    rev = None
    if "@" in model_id:
        model_id, rev = model_id.split("@", 1)

    if is_chrono(model_id):
        mod = _ensure_chrono_module(trainable=trainable)
        # ChronoGPT is GPT-2 BPE; use the HF tokenizer for a uniform interface
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = mod.ChronoGPT.from_pretrained(model_id, cache_dir=None).to(device).eval()

        def fwd(input_ids):
            out = model(input_ids)
            return out[0] if isinstance(out, (tuple, list)) else out

        return tok, fwd, model

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision=rev)
    from transformers import AutoModelForCausalLM
    # transformers renamed torch_dtype to dtype in 4.56, which needs Python 3.9.
    # Three of the hosts ship 3.8 and therefore an older transformers, so the
    # keyword has to follow the installed version rather than be hard coded.
    # from_pretrained swallows unknown kwargs into **kwargs, so the signature
    # cannot be inspected and the version is the only reliable signal.
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    kw = "dtype" if v >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=rev, **{kw: dtype}
    ).to(device).eval()

    def fwd(input_ids):
        return model(input_ids=input_ids).logits

    return tok, fwd, model


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "manelalab/chrono-gpt-v1-20241231"
    tok, fwd, model = load(mid)
    print(f"{mid}: params={n_params(model)/1e9:.3f}B")
    ids = tok("The capital of France is", return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        lg = fwd(ids)
    print("logits", tuple(lg.shape), "dtype", lg.dtype)
    nxt = lg[0, -1].argmax().item()
    print("next token:", repr(tok.decode([nxt])))
