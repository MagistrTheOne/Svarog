#!/usr/bin/env python3
"""SVAROG Stage18H: deterministic verified mathematics curriculum.
No external datasets. Test is never loaded by the trainer.
Splits are assigned by canonical (operation, operands) keys BEFORE templates.
"""
import argparse,collections,hashlib,json,random
from pathlib import Path
from stage3_sft_persona import SVAROG_SYSTEM

SEED=1818
def split_for(key):
 digest=hashlib.sha256((str(SEED)+"|"+repr(key)).encode()).digest()
 n=int.from_bytes(digest[:8],"big")%100
 return "train" if n<80 else "validation" if n<90 else "test"

def step_add(a,b):
 # Always valid, even when a>=10: split b into enough to reach next multiple of ten.
 gap=(10-a%10)%10
 if 0<gap<=b:
  rest=b-gap
  mid=a+gap
  chain=[f"{a} + {b}",f"{a} + {gap} + {rest}",f"{mid} + {rest}",str(a+b)]
  assert a+b==a+gap+rest==mid+rest
  return " = ".join(chain)+"."
 return f"{a} + {b} = {a+b}."

def make_rows():
 # Each task has its own split key; both operand orders belong to same split.
 for a in range(100):
  for b in range(a,100):
   for x,y in ((a,b),) if a==b else ((a,b),(b,a)):
    yield ("add",a,b), "add",x,y,x+y,step_add(x,y)
 for a in range(100):
  for b in range(a+1):
   yield ("sub",a,b),"sub",a,b,a-b,f"{a} - {b} = {a-b}."
 for a in range(13):
  for b in range(a,13):
   for x,y in ((a,b),) if a==b else ((a,b),(b,a)):
    yield ("mul",a,b),"mul",x,y,x*y,f"{x} × {y} = {x*y}."
 for b in range(1,13):
  for quotient in range(13):
   a=b*quotient
   yield ("div",a,b),"div",a,b,quotient,f"{a} ÷ {b} = {quotient}."

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18h_math"))
 p.add_argument("--seed",type=int,default=SEED)
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists; refusing overwrite")
 if a.seed!=SEED:p.error("Seed fixed to 1818 for reproducible splits")
 rng=random.Random(SEED)
 groups={s:[] for s in ("train","validation","test")}
 seen={s:set() for s in groups}
 counts=collections.Counter()
 for key,op,x,y,result,explanation in make_rows():
  split=split_for(key)
  if key in seen[split] and x==y:raise AssertionError("Duplicate canonical key")
  seen[split].add(key)
  assert {"add":lambda:x+y,"sub":lambda:x-y,"mul":lambda:x*y,"div":lambda:x//y}[op]()==result
  if op=="div":assert y>0 and x%y==0
  sym={"add":"+","sub":"-","mul":"×","div":"÷"}[op]
  for mode in ("direct","step"):
   question=(f"Вычисли {x} {sym} {y}." if mode=="direct"
             else f"Вычисли по шагам: {x} {sym} {y}.")
   answer=(str(result)+"." if mode=="direct" else explanation)
   row={"category":op,"mode":mode,"operands":[x,y],"canonical_key":list(key),
        "result":result,"messages":[
         {"role":"system","content":SVAROG_SYSTEM},
         {"role":"user","content":question},
         {"role":"assistant","content":answer}]}
   groups[split].append(row);counts[(split,op,mode)]+=1
 # Ensure disjoint canonical problems, even under operand reversal.
 assert not (seen["train"]&seen["validation"] or seen["train"]&seen["test"] or seen["validation"]&seen["test"])
 a.output.mkdir(parents=True)
 manifest={"seed":SEED,"generator":"stage18h_prepare.py","range_add_sub":[0,99],
           "range_mul":[0,12],"range_divisor":[1,12],
           "split":"SHA256 canonical operation/operand key, 80/10/10",
           "counts":{"/".join(k):v for k,v in sorted(counts.items())},
           "unique_canonical_keys":{s:len(v) for s,v in seen.items()},
           "rules":["train/validation/test have disjoint canonical problems",
                    "both operand orders and both response formats stay in same split",
                    "all targets are checked by Python arithmetic",
                    "no outside dataset; test must never be used to select checkpoints"]}
 (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 for split,rows in groups.items():
  rng.shuffle(rows)
  with (a.output/(split+".jsonl")).open("w",encoding="utf-8") as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
  print(split,len(rows),"records",flush=True)
 print("STAGE18H DATASET VERIFIED",a.output,flush=True)
if __name__=="__main__":main()
