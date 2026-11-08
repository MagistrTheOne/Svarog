#!/usr/bin/env python3
"""Stage18E: exact-answer arithmetic QA on untouched test operand pairs."""
import argparse,json,re,sys
from pathlib import Path
import torch,sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--checkpoint",type=Path,default=Path("stage7_arithmetic/checkpoints/best.pt"))
 p.add_argument("--test",type=Path,default=Path("data_stage18_arithmetic/test.jsonl"))
 p.add_argument("--output",type=Path,default=Path("stage7_arithmetic/test_results.json"))
 p.add_argument("--max-new",type=int,default=16)
 a=p.parse_args()
 if a.output.exists():p.error("Report already exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 if not a.checkpoint.is_file() or not a.test.is_file():p.error("Missing checkpoint or test file")
 rows=[json.loads(line) for line in a.test.open(encoding="utf-8") if line.strip()]
 if len(rows)!=120:p.error("Expected exactly 120 test rows")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
 model.load_state_dict(state["model"],strict=True)
 step=state.get("step")
 del state
 model=model.to("cuda").eval()
 results=[];summary={}
 for row in rows:
  cat=row["category"];q=row["messages"][1]["content"];expected=row["result"]
  answer=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
  # Require a single integer response, with optional trailing period.
  matched=re.fullmatch(r"\s*(-?\d+)\.?\s*",answer)
  correct=bool(matched and int(matched.group(1))==expected)
  results.append({"category":cat,"question":q,"expected":expected,"answer":answer,"correct":correct})
  if cat not in summary:summary[cat]={"correct":0,"total":0}
  summary[cat]["total"]+=1;summary[cat]["correct"]+=int(correct)
 for cat,v in sorted(summary.items()):print(f"{cat}: {v['correct']}/{v['total']}",flush=True)
 score=sum(r["correct"] for r in results)
 print(f"STAGE18E TEST checkpoint_step={step} score={score}/{len(results)}",flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps({"checkpoint":str(a.checkpoint),"step":step,"summary":summary,"score":score,"total":len(results),"results":results},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
