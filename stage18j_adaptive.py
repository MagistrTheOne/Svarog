#!/usr/bin/env python3
"""Stage18J adaptive arithmetic school: prediction -> correction -> training -> validation.
Only phase1 TRAIN is used for mining and supervised corrections. Validation is
used for checkpoint selection. TEST is never opened.
"""
import argparse,collections,json,random,subprocess,sys,re
from pathlib import Path

def load(path):
 return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
def number(s):
 m=re.fullmatch(r"\s*(-?\d+)\s*\.?\s*",s)
 return int(m.group(1)) if m else None
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8") as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def run(cmd):
 print("RUN", " ".join(map(str,cmd)),flush=True)
 subprocess.run([str(x) for x in cmd],check=True)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--source",type=Path,default=Path("data_stage18i_foundations/phase1"))
 p.add_argument("--initial",type=Path,default=Path("stage10_foundations/phase1/step_0150.pt"))
 p.add_argument("--output",type=Path,default=Path("stage11_adaptive"))
 p.add_argument("--rounds",type=int,default=3)
 p.add_argument("--steps-per-round",type=int,default=100)
 p.add_argument("--mine-per-category",type=int,default=60)
 p.add_argument("--replay-per-category",type=int,default=150)
 p.add_argument("--seq",type=int,default=512)
 p.add_argument("--max-new",type=int,default=24)
 p.add_argument("--seed",type=int,default=1811)
 a=p.parse_args()
 if a.rounds<1 or min(a.steps_per_round,a.mine_per_category,a.replay_per_category,a.max_new)<1:p.error("invalid parameters")
 if not a.initial.is_file():p.error("initial checkpoint missing")
 if a.output.exists():p.error("Output already exists; refusing overwrite")
 train=load(a.source/"train.jsonl");validation=load(a.source/"validation.jsonl")
 assert train and validation
 tr_keys={tuple(r["canonical_key"]) for r in train}
 va_keys={tuple(r["canonical_key"]) for r in validation}
 assert tr_keys.isdisjoint(va_keys),"Train/validation leakage"
 cats=("successor","predecessor","compare")
 pools={c:[r for r in train if r["category"]==c] for c in cats}
 if any(not pools[c] for c in cats):p.error("missing training category")
 import torch,sentencepiece as spm
 from stage17_qa import infer
 from stage3_sft_persona import SVAROG_SYSTEM
 if not torch.cuda.is_available():p.error("CUDA required")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 rng=random.Random(a.seed)
 current=a.initial
 history=[]
 for round_no in range(1,a.rounds+1):
  state=torch.load(current,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True);del state
  model.eval()
  errors=collections.defaultdict(list);mining=collections.Counter()
  for cat in cats:
   candidates=pools[cat][:]
   rng.shuffle(candidates)
   for row in candidates[:a.mine_per_category]:
    q=row["messages"][1]["content"]
    gold=number(row["messages"][2]["content"])
    with torch.inference_mode():pred=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
    mining[cat]+=1
    if number(pred)!=gold:errors[cat].append(row)
  print("ROUND",round_no,"MINE",dict(mining),"ERRORS",{c:len(errors[c]) for c in cats},flush=True)
  # Hard examples get extra repetitions, balanced replay prevents catastrophic forgetting.
  dataset=[]
  for cat in cats:
   hard=errors[cat]
   replay=pools[cat][:]
   rng.shuffle(replay)
   selected=replay[:a.replay_per_category]
   dataset.extend(selected)
   dataset.extend(hard*3)
  rng.shuffle(dataset)
  folder=a.output/f"round{round_no}"
  write(folder/"data/train.jsonl",dataset)
  write(folder/"data/validation.jsonl",validation)
  cmd=[sys.executable,"stage18i_train.py","--model-dir",a.model_dir,
       "--stage17-best",current,"--data-dir",folder/"data","--output",folder/"checkpoints",
       "--steps",a.steps_per_round,"--seq",a.seq,"--batch",1,"--accum",16,
       "--lr","2e-6","--eval-every",a.steps_per_round,"--save-every",a.steps_per_round,
       "--val-batches",len(validation),"--seed",a.seed+round_no]
  run(cmd)
  current=folder/"checkpoints"/f"step_{a.steps_per_round:04d}.pt"
  # No test inspection. Validation QA records actual generated predictions.
  state=torch.load(current,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True);del state
  model.eval()
  scores={c:{"correct":0,"total":0} for c in cats}
  details=[]
  for row in validation:
   cat=row["category"];q=row["messages"][1]["content"]
   gold=number(row["messages"][2]["content"])
   with torch.inference_mode():pred=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
   good=number(pred)==gold
   scores[cat]["total"]+=1;scores[cat]["correct"]+=int(good)
   details.append({"category":cat,"question":q,"expected":gold,"answer":pred,"correct":good})
  write(folder/"validation_predictions.jsonl",details)
  print("ROUND",round_no,"VALIDATION",scores,flush=True)
  history.append({"round":round_no,"mined":dict(mining),"errors":{c:len(errors[c]) for c in cats},
                  "validation":scores,"checkpoint":str(current)})
  (a.output/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18J COMPLETE; select checkpoint on per-category validation, not loss",flush=True)
if __name__=="__main__":main()
