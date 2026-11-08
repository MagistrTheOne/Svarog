#!/usr/bin/env python3
"""Stage18A: exact tiny-set memorization evaluation, NOT generalization."""
import argparse,json,sys,math
from pathlib import Path
import torch,sentencepiece as spm
from stage4_train_sft import encode_record,make_batch
from stage17_qa import infer

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--checkpoint",type=Path,default=Path("stage6_tiny/checkpoints/best.pt"))
 p.add_argument("--data",type=Path,default=Path("data_stage18_tiny/train.jsonl"))
 p.add_argument("--output",type=Path,default=Path("stage6_tiny/memorization.json"))
 p.add_argument("--seq",type=int,default=512)
 p.add_argument("--max-new",type=int,default=48)
 a=p.parse_args()
 if a.output.exists():p.error("Report exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 rows=[json.loads(x) for x in a.data.open(encoding="utf-8")]
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
 model.load_state_dict(ckpt["model"],strict=True);del ckpt
 model=model.to("cuda").eval()
 results=[];exact=0;loss_sum=0.;tokens=0
 for row in rows:
  sample=encode_record(row,sp,a.seq)
  x,y=make_batch([sample],[0],sp.pad_id(),a.seq,"cuda")
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
   loss=model(input_ids=x,labels=y)["loss"]
  n=int((y[:,1:]!=-100).sum());loss_sum+=float(loss)*n;tokens+=n
  m=row["messages"];pred=infer(model,sp,m[0]["content"],m[1]["content"],a.max_new)
  expected=m[2]["content"].strip()
  ok=pred.strip()==expected
  exact+=int(ok)
  results.append({"question":m[1]["content"],"expected":expected,"generated":pred,"exact":ok})
  print(("PASS" if ok else "FAIL"),m[1]["content"],"=>",repr(pred),flush=True)
 report={"purpose":"TRAIN-SET MEMORIZATION ONLY","exact":exact,"total":len(rows),"accuracy":exact/len(rows),"assistant_loss":loss_sum/tokens,"items":results}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18A MEMORIZATION",exact,"/",len(rows),"loss",round(loss_sum/tokens,5),"report",a.output,flush=True)
if __name__=="__main__":main()
