"""Measure how retriever.k drives VRAM + time in the ladder's forward path.

Replicates the exact k-dependent stages of
scripts/sweep_marginalized_binary_ladder.sh:
  encoder=rope (D_ehr=128, S_ehr=256)
  retrieval_encoder=token_feature (vocab 151936, D_mem=64), S_doc=256
  fusion=cross_attention_perdoc_medium (d_model=256, heads=8, ff=512, layers=2, p=0.1)
  head=linear in_dim=256 out_dim=N
  marginalized_retrieval=true, mode=binary, similarity=cosine
"""

import json
import sys
import time

import torch
from torch import nn

sys.path.insert(0, "/groups/mm6677_gp/zzw2102/MedRAP/src")

from medrap.model.fusion import PerDocCrossAttentionFusion  # noqa: E402
from medrap.model.heads import LinearHead  # noqa: E402
from medrap.model.model import _marginal_binary_logits  # noqa: E402
from medrap.model.retrieval_encoder import TokenFeatureRetrievalEncoder  # noqa: E402
from medrap.model.retrieval_scoring import differentiable_retrieval_scores  # noqa: E402
from medrap.types import FusionInput, RetrieverOutput  # noqa: E402

DEV = torch.device("cuda")
S_EHR = 256
D_EHR = 128
S_DOC = 256
D_MEM = 64
D_RET = 1024
N_TASKS = 128
VOCAB_DOC = 151936

GB = 1024**3


def build():
    renc = TokenFeatureRetrievalEncoder(vocab_size=VOCAB_DOC, embedding_dim=D_MEM).to(DEV)
    fusion = PerDocCrossAttentionFusion(
        d_model=256, num_heads=8, ff_dim=512, num_layers=2, d_in_patient=D_EHR, d_in_doc=D_MEM, dropout=0.1
    ).to(DEV)
    head = LinearHead(in_dim=256, out_dim=N_TASKS).to(DEV)
    return renc, fusion, head


def step(renc, fusion, head, b, k, backward=True):
    """One fwd(+bwd) of the k-dependent path with tensors of the real shapes."""
    patient_state = torch.randn(b, S_EHR, D_EHR, device=DEV, requires_grad=True)
    query_emb = torch.randn(b, 1, D_RET, device=DEV, requires_grad=True)
    ro = RetrieverOutput(
        doc_tokens=torch.randint(0, VOCAB_DOC, (b, 1, k, S_DOC), device=DEV),
        doc_attention_mask=torch.ones(b, 1, k, S_DOC, dtype=torch.bool, device=DEV),
        doc_key_embeddings=torch.randn(b, 1, k, D_RET, device=DEV),
    )
    mem = renc(ro).retrieval_memory  # (B,1,K,S_doc,D_mem)
    fo = fusion(
        FusionInput(
            patient_state=patient_state,
            retrieval_memory=mem,
            doc_attention_mask=ro.doc_attention_mask,
        )
    )
    fused = fo.fused_state  # (B,K,d_model)
    bb, kk, dd = fused.shape
    per_doc_logits = head(fused.reshape(-1, dd)).view(bb, kk, N_TASKS)
    scores = differentiable_retrieval_scores(query_emb, ro.doc_key_embeddings, similarity="cosine")
    logits = _marginal_binary_logits(per_doc_logits, scores)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits)
    )
    if backward:
        loss.backward()
    return float(loss.detach())


def measure(renc, fusion, head, b, k, backward=True):
    for m in (renc, fusion, head):
        m.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step(renc, fusion, head, b, k, backward)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "peak_alloc_GB": round(peak / GB, 3),
        "peak_reserved_GB": round(reserved / GB, 3),
        "activation_GB": round((peak - base) / GB, 3),
        "sec": round(dt, 4),
    }


def main():
    props = torch.cuda.get_device_properties(0)
    out = {"gpu": props.name, "total_GB": round(props.total_memory / GB, 2), "torch": torch.__version__}
    print(json.dumps(out))

    renc, fusion, head = build()
    # warmup / autotune
    step(renc, fusion, head, 4, 2)
    torch.cuda.synchronize()

    print("\n### train step (fwd+bwd), batch_size=32, scaling in k")
    rows = []
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        try:
            r = measure(renc, fusion, head, 32, k, backward=True)
        except torch.OutOfMemoryError as e:
            r = {"OOM": str(e).split("\n")[0][:150]}
            torch.cuda.empty_cache()
        r["k"] = k
        rows.append(r)
        print(json.dumps(r))

    print("\n### eval step (fwd only, no_grad), batch_size=32")
    for k in (4, 128, 256):
        try:
            with torch.no_grad():
                for m in (renc, fusion, head):
                    m.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                step(renc, fusion, head, 32, k, backward=False)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                print(
                    json.dumps(
                        {
                            "k": k,
                            "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / GB, 3),
                            "activation_GB": round(
                                (torch.cuda.max_memory_allocated() - base) / GB, 3
                            ),
                            "sec": round(dt, 4),
                        }
                    )
                )
        except torch.OutOfMemoryError:
            print(json.dumps({"k": k, "OOM": True}))
            torch.cuda.empty_cache()

    print("\n### max batch_size that fits (fwd+bwd), by k")
    for k in (4, 32, 64, 128, 256):
        best = None
        for b in (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
            try:
                r = measure(renc, fusion, head, b, k, backward=True)
                best = {"k": k, "max_bs": b, **r}
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
        print(json.dumps(best))


if __name__ == "__main__":
    main()
