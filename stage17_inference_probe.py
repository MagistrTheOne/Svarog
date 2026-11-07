#!/usr/bin/env python3
"""Stage17B: read-only generation diagnostics, greedy vs seeded top-p sampling."""
import argparse, json, random, sys
from pathlib import Path
import torch
import sentencepiece as spm
from chat_template_v3 import format_chat
from stage3_sft_persona import SVAROG_SYSTEM
from stage4_qa import TESTS, sha

def run(model,sp,prompt,max_new,mode,seed,temperature,top_p):
    ids=[sp.bos_id()]+sp.encode(prompt,out_type=int)
    if len(ids)+max_new>model.config.max_position_embeddings:
        raise RuntimeError("Prompt too long")
    x=torch.tensor([ids],device="cuda")
    eos=sp.eos_id()
    generated=[]
    rng=torch.Generator(device="cuda").manual_seed(seed)
    finish="max_tokens"
    with torch.inference_mode():
        for _ in range(max_new):
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(input_ids=x)["logits"][0,-1].float()
            if mode=="greedy":
                token=int(logits.argmax())
            else:
                logits=logits/temperature
                sorted_logits,sorted_ids=torch.sort(logits,descending=True)
                probs=torch.softmax(sorted_logits,dim=-1)
                cumulative=torch.cumsum(probs,dim=-1)
                mask=(cumulative-probs)>top_p
                sorted_logits[mask]=-float("inf")
                probs=torch.softmax(sorted_logits,dim=-1)
                choice=torch.multinomial(probs,1,generator=rng)
                token=int(sorted_ids[choice])
            if token==eos:
                finish="eos"
                break
            generated.append(token)
            x=torch.cat((x,torch.tensor([[token]],device="cuda")),dim=1)
            if "[im_end]" in sp.decode(generated):
                finish="im_end"
                break
    raw=sp.decode(generated)
    return {"answer":raw.split("[im_end]")[0].strip(),"finish":finish,"generated_tokens":len(generated),
            "raw_tail":raw[-100:]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
    p.add_argument("--checkpoint",type=Path,default=Path("stage4/checkpoints/best.pt"))
    p.add_argument("--output",type=Path,default=Path("stage4/audit/stage17b.json"))
    p.add_argument("--max-new",type=int,default=96)
    p.add_argument("--temperature",type=float,default=.7)
    p.add_argument("--top-p",type=float,default=.9)
    p.add_argument("--seed",type=int,default=1717)
    a=p.parse_args()
    if a.output.exists():p.error("Refusing overwrite")
    if not torch.cuda.is_available():p.error("CUDA required")
    if not 0<a.top_p<=1 or a.temperature<=0 or a.max_new<1:p.error("Invalid decoding settings")
    before={str(x):sha(x) for x in [a.checkpoint,a.model_dir/"model.safetensors",a.model_dir/"tokenizer.model"]}
    sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
    sys.path.insert(0,str(a.model_dir.resolve()))
    from inference import load_svarog
    model,_=load_svarog(a.model_dir.resolve(),device="cpu")
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(state["model"],strict=True)
    del state
    model=model.to("cuda").eval()
    report={"checkpoint_sha256":before[str(a.checkpoint)],"decoding":{"temperature":a.temperature,"top_p":a.top_p,"seed":a.seed},
            "delimiter_ids":{"bos":sp.bos_id(),"eos":sp.eos_id(),"pad":sp.pad_id(),
            "im_start":sp.encode("[im_start]"),"im_end":sp.encode("[im_end]")},"tests":[]}
    for i,(name,q) in enumerate(TESTS):
        prompt=format_chat([{"role":"system","content":SVAROG_SYSTEM},{"role":"user","content":q}],add_generation_prompt=True)
        greedy=run(model,sp,prompt,a.max_new,"greedy",a.seed+i,a.temperature,a.top_p)
        sample=run(model,sp,prompt,a.max_new,"sample",a.seed+i,a.temperature,a.top_p)
        report["tests"].append({"name":name,"question":q,"prompt_tokens":len(sp.encode(prompt))+1,
                                "greedy":greedy,"sample":sample})
        print(name,"GREEDY:",greedy["answer"],"STOP",greedy["finish"],flush=True)
        print(name,"SAMPLE:",sample["answer"],"STOP",sample["finish"],flush=True)
    after={str(x):sha(x) for x in [a.checkpoint,a.model_dir/"model.safetensors",a.model_dir/"tokenizer.model"]}
    if before!=after:raise RuntimeError("Input file changed")
    report["inputs_unchanged"]=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("STAGE17B COMPLETE",a.output)
if __name__=="__main__":main()
