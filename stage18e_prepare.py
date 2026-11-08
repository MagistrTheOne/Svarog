#!/usr/bin/env python3
"""Stage18E arithmetic isolation: deterministic GPT-6-authored task templates.
Train/validation/test have disjoint operand pairs within each operation.
No Hugging Face datasets. Test is never read by trainer.
"""
import argparse,json,random
from pathlib import Path
from stage3_sft_persona import SVAROG_SYSTEM

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18_arithmetic"))
 a=p.parse_args()
 if a.output.exists():p.error("Output directory already exists; refusing overwrite")
 rng=random.Random(1805)
 groups={"train":[],"validation":[],"test":[]}
 for op,symbol in [("add","+"),("sub","-"),("mul","*")]:
  pairs=[(x,y) for x in range(2,31) for y in range(2,21) if op!="sub" or x>=y]
  rng.shuffle(pairs)
  selections={"train":pairs[:180],"validation":pairs[180:220],"test":pairs[220:260]}
  assert all(len(v)==n for v,n in zip(selections.values(),[180,40,40]))
  for split,pairs_for_split in selections.items():
   for i,(x,y) in enumerate(pairs_for_split):
    result={"add":x+y,"sub":x-y,"mul":x*y}[op]
    templates=[f"Сколько будет {x} {symbol} {y}?",f"Вычисли {x} {symbol} {y}.",f"Calculate {x} {symbol} {y}.",f"Реши: {x} {symbol} {y}."]
    question=templates[i%len(templates)]
    groups[split].append({"category":op,"operands":[x,y],"result":result,
      "messages":[{"role":"system","content":SVAROG_SYSTEM},{"role":"user","content":question},{"role":"assistant","content":str(result)+ "."}]})
 for split,rows in groups.items():
  rng.shuffle(rows)
  assert len({(r["category"],*r["operands"]) for r in rows})==len(rows)
 a.output.mkdir(parents=True)
 for split,rows in groups.items():
  with (a.output/(split+".jsonl")).open("w",encoding="utf-8") as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 print("STAGE18E CORPUS", {k:len(v) for k,v in groups.items()},flush=True)
 print("Each operation has 180 train / 40 validation / 40 untouched test pairs",flush=True)
if __name__=="__main__":main()
