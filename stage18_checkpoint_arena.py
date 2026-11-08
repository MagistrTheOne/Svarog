#!/usr/bin/env python3
"""Stage18D read-only arena: available checkpoints, full 64-record loss and unseen QA.
The Stage18C trainer only saves latest.pt and best.pt, not all intermediate steps.
"""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import torch,sentencepiece as spm
from stage18_personal_train import encode_record,make_batch
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM
from stage18_unseen_qa import TESTS,code_check

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def full_loss(model,rows,sp,seq):
 model.eval()
 total=0.;count=0
 with torch.inference_mode():
  for ids,labels in rows:
   x,y=make_batch([(ids,labels)],[0],sp.pad_id(),seq,"cuda")
   with torch.autocast("cuda",dtype=torch.float16):
    v=model(input_ids=x,labels=y)["loss"]
   n=int((y[:,1:]!=-100).sum().item())
   total+=float(v)*n;count+=n
 return total/count

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--validation",type=Path,default=Path("data_stage18_personal/validation.jsonl"))
 p.add_argument("--output",type=Path,default=Path("stage6_personal/arena_18d.json"))
 p.add_argument("--seq",type=int,default=512)
 p.add_argument("--max-new",type=int,default=48)
 a=p.parse_args()
 if a.output.exists():p.error("Report exists: refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 raw=[json.loads(line) for line in a.validation.open(encoding="utf-8") if line.strip()]
 rows=[encode_record(r,sp,a.seq) for r in raw]
 if any(x is None for x in rows):p.error("Validation sample too long")
 if len(rows)!=64:p.error(f"Expected 64 validation records, found {len(rows)}")
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 candidates=[
 ("STAGE17",Path("stage5/checkpoints/best.pt")),
 ("STAGE18A_TINY",Path("stage6_tiny/checkpoints/best.pt")),
 ("STAGE18C_BEST",Path("stage6_personal/checkpoints/best.pt")),
 ("STAGE18C_LATEST",Path("stage6_personal/checkpoints/latest.pt")),
 ]
 report={"validation_count":len(rows),"test_count":len(TESTS),"notes":["The original Stage18C trainer did not preserve steps 50/100/200/250 individually.","Some unseen QA questions overlap Stage18C training prompts; scores are diagnostic, not a clean holdout."],"checkpoints":[]}
 for label,path in candidates:
  if not path.is_file():
   print("SKIP",label,path,"missing",flush=True);continue
  before=sha(path)
  state=torch.load(path,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True)
  step=state.get("step")
  del state
  loss=full_loss(model,rows,sp,a.seq)
  answers=[]
  for category,q,keys in TESTS:
   answer=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
   passed=code_check(answer) if category=="code" else any(k in answer.casefold() for k in keys)
   answers.append({"category":category,"question":q,"answer":answer,"pass":passed})
   print(label,"PASS" if passed else "FAIL",q,"=>",repr(answer),flush=True)
  if sha(path)!=before:raise RuntimeError("Checkpoint changed during read-only arena")
  result={"name":label,"path":str(path),"sha256":before,"step":step,"full_validation_loss":loss,"qa_pass":sum(x["pass"] for x in answers),"qa_total":len(answers),"answers":answers}
  report["checkpoints"].append(result)
  print("ARENA",label,"step",step,"loss",round(loss,6),"qa",result["qa_pass"],"/",len(answers),flush=True)
 if not report["checkpoints"]:p.error("No checkpoints found")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18D ARENA COMPLETE",a.output,flush=True)
if __name__=="__main__":main()
