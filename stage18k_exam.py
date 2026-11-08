#!/usr/bin/env python3
"""SVAROG Stage18K: frozen mastery exam, baseline vs adaptive checkpoint.
Read-only. Test split used for reporting, NEVER checkpoint selection or training.
"""
import argparse,collections,json,re,sys
from pathlib import Path
import torch
import sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

def parse_number(text):
 m=re.fullmatch(r"\s*(-?\d+)\s*\.?\s*",text)
 return int(m.group(1)) if m else None

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--data-dir",type=Path,default=Path("data_stage18i_foundations/phase1"))
 p.add_argument("--baseline",type=Path,default=Path("stage10_foundations/phase1/step_0150.pt"))
 p.add_argument("--adaptive",type=Path,default=Path("stage11_adaptive/round3/checkpoints/step_0100.pt"))
 p.add_argument("--output",type=Path,default=Path("stage12_mastery/exam.json"))
 p.add_argument("--max-new",type=int,default=32)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 if a.max_new<1:p.error("max-new must be positive")
 for f in (a.baseline,a.adaptive,a.data_dir/"test.jsonl"):
  if not f.is_file():p.error(f"Missing {f}")
 rows=[json.loads(x) for x in (a.data_dir/"test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
 train_keys={tuple(json.loads(x)["canonical_key"]) for x in (a.data_dir/"train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
 val_keys={tuple(json.loads(x)["canonical_key"]) for x in (a.data_dir/"validation.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
 test_keys={tuple(x["canonical_key"]) for x in rows}
 if train_keys & test_keys or val_keys & test_keys:p.error("Canonical split leakage")
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 model=model.to("cuda").eval()
 # Additional formats are derived from held-out test keys ONLY, no training use.
 variants={
  "successor":lambda n:f"Назови число, следующее за {n}.",
  "predecessor":lambda n:f"Назови число, предшествующее {n}.",
  "compare":lambda n:f"Из двух чисел {n[0]} и {n[1]} назови большее."
 }
 dialogue=[
  "Привет! Как тебя зовут?",
  "Расскажи в двух предложениях, что такое искусственный интеллект.",
  "Объясни, почему нужно проверять результаты эксперимента.",
  "Что ты умеешь, а чего не умеешь?",
  "Напиши короткое дружелюбное приветствие."
 ]
 report={"protocol":"stage18k-v1","split":"test","test_records":len(rows),"checkpoints":{}}
 for label,ckpt in (("baseline",a.baseline),("adaptive",a.adaptive)):
  state=torch.load(ckpt,map_location="cpu",weights_only=False)
  model.load_state_dict(state["model"],strict=True)
  del state
  model.eval()
  scores=collections.defaultdict(lambda:{"correct":0,"total":0})
  details=[]
  for row in rows:
   cat=row["category"];q=row["messages"][1]["content"]
   expected=parse_number(row["messages"][2]["content"])
   with torch.inference_mode():response=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
   parsed=parse_number(response)
   correct=parsed==expected
   scores[cat]["total"]+=1;scores[cat]["correct"]+=int(correct)
   details.append({"category":cat,"format":"original","question":q,"expected":expected,"response":response,"parsed":parsed,"correct":correct})
   if cat in ("successor","predecessor"):
    n=int(row["canonical_key"][1]);q2=variants[cat](n)
   else:
    n1,n2=map(int,row["canonical_key"][1:]);q2=variants[cat]((n1,n2))
   with torch.inference_mode():response2=infer(model,sp,SVAROG_SYSTEM,q2,a.max_new)
   parsed2=parse_number(response2)
   k=cat+"_paraphrase"
   scores[k]["total"]+=1;scores[k]["correct"]+=int(parsed2==expected)
   details.append({"category":cat,"format":"paraphrase","question":q2,"expected":expected,"response":response2,"parsed":parsed2,"correct":parsed2==expected})
  chat=[]
  for question in dialogue:
   with torch.inference_mode():answer=infer(model,sp,SVAROG_SYSTEM,question,96)
   chat.append({"question":question,"answer":answer})
  report["checkpoints"][label]={"path":str(ckpt),"scores":dict(scores),"predictions":details,"dialogue":chat}
  print("EXAM",label,dict(scores),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18K EXAM COMPLETE",a.output,flush=True)
 print("WARNING: TEST IS SEALED AFTER THIS RUN. Do not use it for hyperparameter or checkpoint selection.",flush=True)
if __name__=="__main__":main()
