#!/usr/bin/env python3
"""Stage18I exact-answer evaluation. Never used by trainer."""
import argparse,json,re,sys,collections
from pathlib import Path
import torch,sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

def final_number(text):
 m=re.search(r"(-?\d+)\s*\.?\s*$",text.strip())
 return int(m.group(1)) if m else None

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--data-dir",type=Path,default=Path("data_stage18i_math"))
 p.add_argument("--checkpoint-dir",type=Path,default=Path("stage10_foundations/phase1"))
 p.add_argument("--output",type=Path,default=Path("stage10_foundations/phase1_qa.json"))
 p.add_argument("--split",choices=["validation","test"],default="test")
 p.add_argument("--per-category",type=int,default=30)
 p.add_argument("--max-new",type=int,default=64)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 if a.per_category<1:p.error("per-category must be positive")
 checkpoints=sorted(a.checkpoint_dir.glob("step_*.pt"))
 if not checkpoints:p.error("No step checkpoints")
 rows=[json.loads(line) for line in (a.data_dir/(a.split+".jsonl")).open(encoding="utf-8")]
 chosen=[];counts=collections.Counter()
 for row in rows:
  key=(row["category"],row["mode"])
  if counts[key]<a.per_category:
   chosen.append(row);counts[key]+=1
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 report=[]
 for path in checkpoints:
  state=torch.load(path,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True)
  step=state.get("step");del state
  scores=collections.defaultdict(lambda:{"correct":0,"total":0})
  for row in chosen:
   q=row["messages"][1]["content"]
   answer=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
   result=final_number(answer)
   key=row["category"]+"/"+row["mode"]
   scores[key]["total"]+=1
   scores[key]["correct"]+=int(result==row["result"])
   report.append({"step":step,"category":row["category"],"mode":row["mode"],
                  "operands":row["operands"],"expected":row["result"],
                  "answer":answer,"parsed":result,"correct":result==row["result"]})
  print("QA",step,a.split,dict(scores),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps({"split":a.split,"per_category":a.per_category,"results":report},ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18I QA COMPLETE",a.output,flush=True)
if __name__=="__main__":main()
