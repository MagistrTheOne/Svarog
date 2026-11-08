#!/usr/bin/env python3
"""SVAROG Stage18I: staged arithmetic foundations, deterministic and disjoint.
Training levels: counting/comparison -> single digit addition -> two digit addition.
Test data is never consumed by trainer; run QA on validation to select checkpoint.
"""
import argparse,collections,hashlib,json,random
from pathlib import Path
from stage3_sft_persona import SVAROG_SYSTEM

def split(key):
 n=int.from_bytes(hashlib.sha256(("1819|"+repr(key)).encode()).digest()[:8],"big")%100
 return "train" if n<80 else "validation" if n<90 else "test"

def tasks():
 for n in range(0,100):
  yield 1,("successor",n),f"Какое число идёт после {n}?",f"{n+1}.","successor"
  if n>0:yield 1,("predecessor",n),f"Какое число идёт перед {n}?",f"{n-1}.","predecessor"
 for a in range(0,30):
  for b in range(a+1,30):
   yield 1,("compare",a,b),f"Какое число больше: {a} или {b}?",f"{b}.","compare"
   yield 1,("compare",a,b),f"Какое число больше: {b} или {a}?",f"{b}.","compare"
 for a in range(10):
  for b in range(a,10):
   for x,y in ([(a,b)] if a==b else [(a,b),(b,a)]):
    yield 2,("add1",a,b),f"Вычисли {x} + {y}.",f"{x+y}.","add1"
 for a in range(0,100):
  for b in range(a,100):
   if a<10 and b<10:continue
   for x,y in ([(a,b)] if a==b else [(a,b),(b,a)]):
    yield 3,("add2",a,b),f"Вычисли {x} + {y}.",f"{x+y}.","add2"

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18i_foundations"))
 a=p.parse_args()
 if a.output.exists():p.error("Output exists, refusing overwrite")
 rng=random.Random(1819)
 data={phase:{s:[] for s in ("train","validation","test")} for phase in (1,2,3)}
 keys={phase:{s:set() for s in ("train","validation","test")} for phase in (1,2,3)}
 for phase,key,question,answer,category in tasks():
  s=split(key)
  keys[phase][s].add(key)
  row={"category":category,"canonical_key":list(key),"messages":[
   {"role":"system","content":SVAROG_SYSTEM},
   {"role":"user","content":question},
   {"role":"assistant","content":answer}]}
  data[phase][s].append(row)
 for phase in (1,2,3):
  sets=keys[phase]
  assert not (sets["train"]&sets["validation"] or sets["train"]&sets["test"] or sets["validation"]&sets["test"])
  folder=a.output/f"phase{phase}"
  folder.mkdir(parents=True,exist_ok=True)
  for s,rows in data[phase].items():
   rng.shuffle(rows)
   with (folder/(s+".jsonl")).open("w",encoding="utf-8") as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
   print(f"PHASE {phase} {s}: {len(rows)} records; {len(sets[s])} canonical tasks",flush=True)
 manifest={"seed":1819,"phase1":"successor, predecessor, comparison","phase2":"addition operands 0..9",
           "phase3":"addition operands 0..99, excluding phase2",
           "split":"SHA256 canonical keys 80/10/10; operand reversals grouped",
           "warning":"Do not select checkpoint on test. Use validation QA; test once at end."}
 (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18I DATASET VERIFIED",a.output,flush=True)
if __name__=="__main__":main()
