#!/usr/bin/env python3
"""NULLXES SVAROG | Stage18G read-only arithmetic failure analysis.
No training, no dataset mutation, no checkpoint mutation.
"""
import argparse,collections,hashlib,json,re,sys
from pathlib import Path
import torch
import sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def numeric_answer(text):
 match=re.fullmatch(r"\s*(-?\d+)\.?\s*",text)
 return int(match.group(1)) if match else None

def equations(text):
 chunks=[s.strip() for s in text.strip().rstrip(".").split("=")]
 if len(chunks)<2:return {"has_equation":False,"valid":False,"final":None,"count":0}
 vals=[]
 for chunk in chunks:
  if not re.fullmatch(r"\d+(?:\s*\+\s*\d+)*",chunk):
   return {"has_equation":True,"valid":False,"final":None,"count":len(chunks)}
  vals.append(sum(int(n) for n in re.findall(r"\d+",chunk)))
 return {"has_equation":True,"valid":len(chunks)>=3 and len(set(vals))==1,"final":vals[-1],"count":len(chunks)}

def token_report(sp):
 items={}
 for label,values in (
  ("numbers_0_99",[str(i) for i in range(100)]),
  ("expressions",["7 + 8","8 + 7","9 + 4","12 + 7","19 + 19","Вычисли 7 + 8.","Вычисли по шагам: 7 + 8."])):
  records=[]
  for value in values:
   ids=sp.encode(value,out_type=int)
   pieces=sp.encode(value,out_type=str)
   records.append({"text":value,"ids":ids,"pieces":pieces,"tokens":len(ids)})
  items[label]=records
 return items

def get_pairs(path):
 rows=[json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
 pairs={}
 for row in rows:
  key=tuple(row["pair"])
  pairs.setdefault(key,row)
 return pairs

def characterize(records):
 errors=collections.Counter(); offsets=collections.Counter();outputs=collections.Counter()
 for row in records:
  if row["correct"]:continue
  value=row["numeric"]
  if value is None:
   errors["non_numeric"]+=1
   continue
  outputs[str(value)]+=1
  delta=value-row["expected"]
  offsets[str(delta)]+=1
  if abs(delta)==1:errors["off_by_one"]+=1
  elif abs(delta)<=3:errors["off_by_2_or_3"]+=1
  else:errors["other_numeric"]+=1
 return {"error_types":dict(errors),"most_common_wrong_numbers":outputs.most_common(10),"most_common_offsets":offsets.most_common(10)}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--data-dir",type=Path,default=Path("data_stage18f_addition"))
 p.add_argument("--direct",type=Path,default=Path("stage8_addition/checkpoints/direct/step_0200.pt"))
 p.add_argument("--step",type=Path,default=Path("stage8_addition/checkpoints/step/step_0200.pt"))
 p.add_argument("--output",type=Path,default=Path("stage8_addition/diagnostics_18g.json"))
 p.add_argument("--limit",type=int,default=0,help="Canonical test pairs; 0=all 40")
 p.add_argument("--max-new",type=int,default=64)
 p.add_argument("--step-lengths",type=int,nargs="+",default=[16,32,64,96])
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 if a.limit<0 or a.max_new<1 or not a.step_lengths or any(v<1 for v in a.step_lengths):p.error("Invalid generation settings")
 inputs=[a.model_dir/"tokenizer.model",a.model_dir/"model.safetensors",a.direct,a.step,
         a.data_dir/"direct_train.jsonl",a.data_dir/"direct_test.jsonl"]
 for path in inputs:
  if not path.is_file():p.error(f"Missing {path}")
 hashes={str(path):sha(path) for path in inputs}
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 tokens=token_report(sp)
 pairs=get_pairs(a.data_dir/"direct_test.jsonl")
 train=get_pairs(a.data_dir/"direct_train.jsonl")
 if set(pairs)&set(train):p.error("Train/test operand leakage")
 if len(pairs)!=40:p.error(f"Expected 40 test pairs, found {len(pairs)}")
 if a.limit:pairs=dict(list(pairs.items())[:a.limit])
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 report={"stage":"18G","mode":"read_only","tokenizer":tokens,
         "test_unique_pairs":len(pairs),"checkpoints":{},"sha256_before":hashes}
 for arm,path in (("direct",a.direct),("step",a.step)):
  state=torch.load(path,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True)
  step_num=state.get("step");del state
  print("CHECKPOINT",arm,"step",step_num,flush=True)
  results=[];commutativity=[]
  for (x,y),_ in pairs.items():
   variants=[(x,y)] if x==y else [(x,y),(y,x)]
   per_pair=[]
   for left,right in variants:
    expected=left+right
    prompt=f"Вычисли {left} + {right}." if arm=="direct" else f"Вычисли по шагам: {left} + {right}."
    lengths=[a.max_new] if arm=="direct" else sorted(set(a.step_lengths))
    for max_new in lengths:
     answer=infer(model,sp,SVAROG_SYSTEM,prompt,max_new)
     num=numeric_answer(answer)
     eq=equations(answer)
     final=num if arm=="direct" else eq["final"]
     correct=final==expected
     row={"pair":[x,y],"operands":[left,right],"carry":(left%10+right%10)>=10,
          "expected":expected,"max_new":max_new,"answer":answer,"numeric":num,
          "final":final,"correct":correct,"verified_equations":eq["valid"],
          "equation_parts":eq["count"]}
     results.append(row)
     if max_new==max(lengths):per_pair.append(row)
   if len(per_pair)==2:
    a1,a2=per_pair
    commutativity.append({"pair":[x,y],"same_text":a1["answer"]==a2["answer"],
       "same_final":a1["final"] is not None and a1["final"]==a2["final"],
       "both_correct":a1["correct"] and a2["correct"],
       "forward":a1["answer"],"reverse":a2["answer"]})
  stats={}
  for length in sorted({r["max_new"] for r in results}):
   chosen=[r for r in results if r["max_new"]==length]
   for carry in (False,True):
    group=[r for r in chosen if r["carry"]==carry]
    stats[f"length={length};carry={carry}"]={"correct":sum(r["correct"] for r in group),
      "total":len(group),"valid_equations":sum(r["verified_equations"] for r in group),
      "error_analysis":characterize(group)}
   stats[f"length={length};all"]={"correct":sum(r["correct"] for r in chosen),
     "total":len(chosen),"valid_equations":sum(r["verified_equations"] for r in chosen)}
  report["checkpoints"][arm]={"checkpoint":str(path),"step":step_num,"stats":stats,
    "commutativity":{"pairs":len(commutativity),
      "same_text":sum(r["same_text"] for r in commutativity),
      "same_final":sum(r["same_final"] for r in commutativity),
      "both_correct":sum(r["both_correct"] for r in commutativity),
      "examples":commutativity},"raw":results}
  for name,val in stats.items():
   if name.endswith(";all"):print("SUMMARY",arm,name,val,flush=True)
  print("COMMUTATIVITY",arm,report["checkpoints"][arm]["commutativity"]["same_final"],
        "/",len(commutativity),"same numeric results",flush=True)
 after={str(path):sha(path) for path in inputs}
 if hashes!=after:raise RuntimeError("Read-only inputs changed during audit")
 report["inputs_unchanged"]=True
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18G COMPLETE",a.output,flush=True)
if __name__=="__main__":main()
