#!/usr/bin/env python3
"""Stage18F checkpoint audit: familiar/unseen pairs, direct/step prompts.
Checks numeric final result and every equality in a step response.
"""
import argparse,json,re,sys
from pathlib import Path
import torch,sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

def math_value(expr):
 if not re.fullmatch(r"\s*\d+(?:\s*\+\s*\d+)*\s*",expr):return None
 return sum(map(int,re.findall(r"\d+",expr)))

def judge(answer,expected,mode):
 cleaned=answer.strip().rstrip(".").strip()
 if mode=="direct":
  m=re.fullmatch(r"(\d+)",cleaned)
  return bool(m and int(m.group(1))==expected),False
 chunks=[s.strip() for s in cleaned.split("=")]
 values=[math_value(s) for s in chunks]
 final_correct=bool(len(chunks)>=2 and values[-1]==expected)
 verified=bool(len(chunks)>=3 and all(v==expected for v in values))
 return final_correct,verified

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--checkpoint-dir",type=Path,default=Path("stage8_addition/checkpoints/direct"))
 p.add_argument("--data-dir",type=Path,default=Path("data_stage18f_addition"))
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--limit",type=int,default=30,help="Canonical pairs per split; 0 means all")
 p.add_argument("--max-new",type=int,default=64)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 if a.limit<0 or a.max_new<1:p.error("Invalid limit/max-new")
 paths=sorted(a.checkpoint_dir.glob("step_*.pt"))
 if not paths:p.error("No retained checkpoints found")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 # Read each split once, one canonical order per pair to avoid correlated duplicate samples.
 samples={}
 for split in ("train","test"):
  file=a.data_dir/f"direct_{split}.jsonl"
  rows=[json.loads(line) for line in file.open(encoding="utf-8") if line.strip()]
  seen=set();chosen=[]
  for row in rows:
   pair=tuple(row["pair"])
   if pair in seen:continue
   seen.add(pair);chosen.append(row)
   if a.limit and len(chosen)>=a.limit:break
  samples[split]=chosen
 print("STAGE18F QA", {k:len(v) for k,v in samples.items()},"pairs per mode",flush=True)
 results=[]
 for path in paths:
  state=torch.load(path,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True)
  step=state.get("step");del state
  for split,rows in samples.items():
   for mode in ("direct","step"):
    correct=0;verified=0;details=[]
    for row in rows:
     x,y=row["operands"];expected=x+y
     question=f"Вычисли {x} + {y}." if mode=="direct" else f"Вычисли по шагам: {x} + {y}."
     answer=infer(model,sp,SVAROG_SYSTEM,question,a.max_new)
     ok,proof=judge(answer,expected,mode)
     correct+=int(ok);verified+=int(proof)
     details.append({"pair":[x,y],"answer":answer,"expected":expected,"correct":ok,"verified_steps":proof})
    item={"checkpoint":path.name,"step":step,"split":split,"prompt_mode":mode,
          "correct":correct,"total":len(rows),"verified_steps":verified,"details":details}
    results.append(item)
    print(f"QA step={step} {split} {mode}: {correct}/{len(rows)} valid_steps={verified}/{len(rows)}",flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps({"checkpoint_dir":str(a.checkpoint_dir),"limit":a.limit,"results":results},ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18F QA COMPLETE",a.output,flush=True)
if __name__=="__main__":main()
