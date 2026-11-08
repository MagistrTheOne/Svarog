#!/usr/bin/env python3
"""Stage18L adaptive training: inference -> error mining -> supervised correction -> QA.
No test access. Numeric train/validation keys disjoint; validation phrasing held out.
"""
import argparse,collections,json,random,subprocess,sys,re
from pathlib import Path
import torch,sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

CATS=("successor","predecessor","compare")
def load(p):
 return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def num(s):
 m=re.fullmatch(r"\s*(-?\d+)\s*\.?\s*",s)
 return int(m.group(1)) if m else None
def save(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8") as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--data-dir",type=Path,default=Path("data_stage18l_concepts"))
 p.add_argument("--initial",type=Path,default=Path("stage11_adaptive/round3/checkpoints/step_0100.pt"))
 p.add_argument("--output",type=Path,default=Path("stage13_concepts"))
 p.add_argument("--rounds",type=int,default=3)
 p.add_argument("--steps",type=int,default=100)
 p.add_argument("--mine",type=int,default=45)
 p.add_argument("--replay",type=int,default=150)
 p.add_argument("--max-new",type=int,default=24)
 p.add_argument("--seed",type=int,default=1813)
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists")
 if not a.initial.is_file():p.error("Initial checkpoint missing")
 if min(a.rounds,a.steps,a.mine,a.replay,a.max_new)<1:p.error("Invalid hyperparameters")
 if not torch.cuda.is_available():p.error("CUDA required")
 train=load(a.data_dir/"train.jsonl")
 val=load(a.data_dir/"validation.jsonl")
 para=load(a.data_dir/"validation_paraphrase.jsonl")
 train_keys={tuple(r["canonical_key"]) for r in train}
 val_keys={tuple(r["canonical_key"]) for r in val+para}
 if train_keys&val_keys:p.error("Canonical train/validation leakage")
 if any(r["template_id"].startswith("held_") for r in train):p.error("Held-out template in train")
 pools={c:[r for r in train if r["category"]==c] for c in CATS}
 if any(not pools[c] for c in CATS):p.error("Missing category")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 rng=random.Random(a.seed)
 current=a.initial
 history=[]
 for rnd in range(1,a.rounds+1):
  state=torch.load(current,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True);del state
  model.eval()
  hard={};mine_scores={}
  for cat in CATS:
   candidates=pools[cat][:]
   rng.shuffle(candidates)
   sample=candidates[:a.mine]
   mistakes=[]
   for row in sample:
    q=row["messages"][1]["content"];gold=num(row["messages"][2]["content"])
    with torch.inference_mode():answer=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
    if num(answer)!=gold:mistakes.append(row)
   hard[cat]=mistakes
   mine_scores[cat]={"wrong":len(mistakes),"total":len(sample)}
  print("ROUND",rnd,"PREDICT",mine_scores,flush=True)
  # Equal total sample weight per skill, mixing error corrections and replay.
  corpus=[]
  for cat in CATS:
   replay=pools[cat][:]
   rng.shuffle(replay)
   selected=replay[:a.replay]
   mistakes=hard[cat]
   correction=mistakes*3
   candidates=selected+correction
   if not candidates:raise RuntimeError("Empty skill pool")
   corpus.extend(rng.choices(candidates,k=a.replay))
  rng.shuffle(corpus)
  folder=a.output/f"round{rnd}"
  save(folder/"data/train.jsonl",corpus)
  # Trainer requires validation.jsonl; evaluation loss uses ordinary validation only.
  save(folder/"data/validation.jsonl",val)
  cmd=[sys.executable,"stage18i_train.py","--model-dir",str(a.model_dir),
       "--stage17-best",str(current),"--data-dir",str(folder/"data"),
       "--output",str(folder/"checkpoints"),"--steps",str(a.steps),
       "--seq","512","--batch","1","--accum","16","--lr","2e-6",
       "--eval-every",str(a.steps),"--save-every",str(a.steps),
       "--val-batches",str(len(val)),"--seed",str(a.seed+rnd)]
  print("RUN"," ".join(cmd),flush=True)
  subprocess.run(cmd,check=True)
  current=folder/"checkpoints"/f"step_{a.steps:04d}.pt"
  state=torch.load(current,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True);del state
  model.eval()
  scores={}
  for name,rows in (("seen_wording",val),("unseen_wording",para)):
   tally=collections.defaultdict(lambda:{"correct":0,"total":0})
   details=[]
   for row in rows:
    cat=row["category"];q=row["messages"][1]["content"]
    gold=num(row["messages"][2]["content"])
    with torch.inference_mode():response=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
    ok=num(response)==gold
    tally[cat]["correct"]+=int(ok);tally[cat]["total"]+=1
    details.append({"category":cat,"question":q,"expected":gold,"answer":response,"correct":ok})
   save(folder/(name+"_predictions.jsonl"),details)
   scores[name]=dict(tally)
  print("ROUND",rnd,"VALIDATION",json.dumps(scores,ensure_ascii=False),flush=True)
  history.append({"round":rnd,"mining":mine_scores,"validation":scores,"checkpoint":str(current)})
  a.output.mkdir(parents=True,exist_ok=True)
  (a.output/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18L COMPLETE. No test was opened.",flush=True)
if __name__=="__main__":main()
