#!/usr/bin/env python3
"""Stage18F: deterministic 0..19 addition curriculum, two independent arms.
Disjoint unordered operand pairs prevent reversed-pair leakage.
"""
import argparse,json,random
from pathlib import Path
from stage3_sft_persona import SVAROG_SYSTEM

def explanation(a,b):
 # Split b to reach the next ten from a; for a>=10, use direct decomposition.
 if a<10 and a+b>=10 and b>0:
  first=10-a;rest=b-first
  return f"{a} + {b} = {a} + {first} + {rest} = 10 + {rest} = {a+b}."
 return f"{a} + {b} = {a+b}."

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18f_addition"))
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists; refusing overwrite")
 rng=random.Random(1806)
 # Canonical unordered pairs; 210 unique pairs, 130 train / 40 val / 40 test.
 pairs=[(x,y) for x in range(20) for y in range(x,20)]
 rng.shuffle(pairs)
 splits={"train":pairs[:130],"validation":pairs[130:170],"test":pairs[170:210]}
 a.output.mkdir(parents=True)
 manifest={"seed":1806,"range":[0,19],"pair_policy":"unordered pairs disjoint across splits","counts":{k:len(v) for k,v in splits.items()}}
 (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 for split,items in splits.items():
  for arm in ("direct","step"):
   rows=[]
   for x,y in items:
    # Train both orders; test records contain both orders to expose asymmetry.
    for left,right in ((x,y), (y,x)) if x!=y else ((x,y),):
     q=f"Вычисли {left} + {right}."
     ans=(str(left+right)+".") if arm=="direct" else explanation(left,right)
     rows.append({"pair":[x,y],"operands":[left,right],"mode":arm,
      "messages":[{"role":"system","content":SVAROG_SYSTEM},
                  {"role":"user","content":q if arm=="direct" else f"Вычисли по шагам: {left} + {right}."},
                  {"role":"assistant","content":ans}]})
   with (a.output/f"{arm}_{split}.jsonl").open("w",encoding="utf-8") as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
   print(f"{arm} {split}: {len(rows)} prompts from {len(items)} unordered pairs",flush=True)
 print("STAGE18F PREP COMPLETE",a.output,flush=True)
if __name__=="__main__":main()
