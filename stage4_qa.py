#!/usr/bin/env python3
"""Stage 16: read-only, reproducible HF Base / Stage13 / Stage15 assistant QA."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
import sentencepiece as spm
from stage4_train_sft import encode_record, make_batch

TESTS = [
    ("RU identity", "Кто ты? Представься одним предложением."),
    ("RU factual", "Какая столица Франции? Ответь кратко."),
    ("RU arithmetic", "У Маши 7 яблок. Она отдала 3 и купила 2. Сколько осталось?"),
    ("RU logic", "Все кошки — животные. Все животные дышат. Следует ли, что кошки дышат?"),
    ("RU code", "Напиши функцию Python square(x), которая возвращает квадрат x."),
    ("RU uncertainty", "Какой пароль от моего домашнего Wi-Fi?"),
    ("RU explanation", "Объясни простыми словами, что такое градиентный спуск."),
    ("EN factual", "What is the capital of Germany? Answer briefly."),
    ("EN code", "Write a Python function that returns the larger of two numbers."),
    ("EN uncertainty", "What is the serial number of my laptop?"),
]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def infer(model, sp, system, question, max_new):
    prefix=f"[im_start]system\\n{system}[im_end]\\n[im_start]user\\n{question}[im_end]\\n[im_start]assistant\\n"
    ids=[sp.bos_id()]+sp.encode(prefix,out_type=int)
    if len(ids)+max_new>model.config.max_position_embeddings:
        raise RuntimeError("Prompt exceeds context")
    x=torch.tensor([ids],device="cuda",dtype=torch.long)
    generated=[]
    eos=sp.eos_id()
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new):
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(input_ids=x)["logits"][0,-1].float()
            tok=int(torch.argmax(logits))
            if tok==eos:
                break
            generated.append(tok)
            x=torch.cat([x,torch.tensor([[tok]],device="cuda")],dim=1)
            if "[im_end]" in sp.decode(generated):
                break
    return sp.decode(generated).split("[im_end]")[0].strip()

def validation_loss(model, samples, pad, seq, limit):
    model.eval()
    total=0.0
    count=0
    with torch.inference_mode():
        for i in range(min(limit,len(samples))):
            x,y=make_batch(samples,[i],pad,seq,"cuda")
            with torch.autocast("cuda",dtype=torch.float16):
                v=model(input_ids=x,labels=y)["loss"]
            n=int((y[:,1:]!=-100).sum())
            if not math.isfinite(float(v)):
                raise RuntimeError("Nonfinite loss")
            total+=float(v)*n
            count+=n
    return total/count

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
    p.add_argument("--stage13-best",type=Path,default=Path("stage3/checkpoints/best.pt"))
    p.add_argument("--stage15-best",type=Path,default=Path("stage4/checkpoints/best.pt"))
    p.add_argument("--validation",type=Path,default=Path("data_stage3_sft/validation.jsonl"))
    p.add_argument("--output",type=Path,default=Path("stage4/qa_stage16.json"))
    p.add_argument("--seq",type=int,default=512)
    p.add_argument("--val-batches",type=int,default=128)
    p.add_argument("--max-new",type=int,default=96)
    a=p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required")
    if min(a.seq,a.val_batches,a.max_new)<1:
        p.error("Invalid QA settings")
    if a.output.exists():
        p.error("Report exists: refusing overwrite")
    paths=[a.model_dir/"model.safetensors",a.model_dir/"tokenizer.model",a.stage13_best,a.stage15_best,a.validation]
    for path in paths:
        if not path.is_file():
            p.error(f"Missing {path}")
    before={str(path):sha(path) for path in paths}
    sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
    samples=[]
    with a.validation.open(encoding="utf-8") as f:
        for line in f:
            item=encode_record(json.loads(line),sp,a.seq)
            if item is not None:
                samples.append(item)
    if not samples:
        p.error("No validation records fit")
    sys.path.insert(0,str(a.model_dir.resolve()))
    from inference import load_svarog
    from stage3_sft_persona import SVAROG_SYSTEM
    model,_=load_svarog(a.model_dir.resolve(),device="cpu")
    model=model.to("cuda").eval()
    if a.seq>model.config.max_position_embeddings:
        p.error("Sequence exceeds context")
    stages=[("HF_BASE",None),("STAGE13",a.stage13_best),("STAGE15",a.stage15_best)]
    report={"status":"STARTED","validation":"Stage14 held-out assistant-only","validation_records_available":len(samples),
            "validation_records_evaluated":min(a.val_batches,len(samples)),"decoding":"greedy argmax",
            "max_new":a.max_new,"seq":a.seq,"sha256":before,"stages":{}}
    print("NULLXES SVAROG | STAGE 16 | READ-ONLY THREE-WAY QA",flush=True)
    for label,ckpt in stages:
        if ckpt is not None:
            state=torch.load(ckpt,map_location="cpu",weights_only=False)
            if label=="STAGE15":
                fp=state.get("fingerprint",{})
                if fp.get("stage13_sha256")!=before[str(a.stage13_best)]:
                    raise RuntimeError("Stage15 checkpoint not derived from supplied Stage13")
                if fp.get("validation_sha256")!=before[str(a.validation)]:
                    raise RuntimeError("Stage15 validation fingerprint mismatch")
                if fp.get("tokenizer_sha256")!=before[str(a.model_dir/"tokenizer.model")]:
                    raise RuntimeError("Stage15 tokenizer fingerprint mismatch")
                if fp.get("seq")!=a.seq:
                    raise RuntimeError("Stage15 sequence length mismatch")
            model.load_state_dict(state["model"],strict=True)
            step=state.get("step")
            del state
        else:
            step=None
        loss=validation_loss(model,samples,sp.pad_id(),a.seq,a.val_batches)
        print(f"{label} step={step} assistant_loss={loss:.5f} ppl={math.exp(min(loss,30)):.2f}",flush=True)
        outputs=[]
        for name,q in TESTS:
            ans=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
            outputs.append({"test":name,"question":q,"answer":ans})
            print(f"  [{name}] {ans}",flush=True)
        report["stages"][label]={"step":step,"assistant_loss":loss,"ppl":math.exp(min(loss,30)),"qa":outputs}
        torch.cuda.empty_cache()
    after={str(path):sha(path) for path in paths}
    if before!=after:
        raise RuntimeError("Source files changed during read-only QA")
    report["status"]="COMPLETE"
    report["inputs_unchanged"]=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix(".json.partial")
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(a.output)
    print("STAGE 16 QA COMPLETE:",a.output,flush=True)
if __name__=="__main__":
    main()
