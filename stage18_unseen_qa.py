#!/usr/bin/env python3
"""Stage18B: read-only unseen prompt probe, checkpoint unchanged."""
import argparse,json,sys,hashlib,ast
from pathlib import Path
import torch,sentencepiece as spm
from stage17_qa import infer
from stage3_sft_persona import SVAROG_SYSTEM

TESTS=[
("paraphrase","Назови столицу Франции.",["париж"]),
("paraphrase","Как тебя зовут?",["сварог"]),
("paraphrase","Где находится столица Германии? Назови город.",["берлин"]),
("arithmetic","Сколько будет 8 + 5?",["13"]),
("arithmetic","Сколько будет 9 * 3?",["27"]),
("arithmetic","What is 6 + 7?",["13"]),
("arithmetic","Сколько будет 14 - 6?",["8"]),
("arithmetic","What is 4 times 6?",["24"]),
("uncertainty","Можешь назвать мой секретный PIN-код?",["не знаю","нет данных","неизвест","не могу знать"]),
("code","Напиши функцию Python cube(x), возвращающую куб числа.",None),
]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def code_check(answer):
 try:
  tree=ast.parse(answer)
  funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="cube" and len(x.args.args)==1]
  if not funcs:return False
  # Strict structural check, no execution of generated code.
  fn=funcs[0]
  if len(fn.body)!=1 or not isinstance(fn.body[0],ast.Return):return False
  e=fn.body[0].value
  def is_x(n):return isinstance(n,ast.Name) and n.id==fn.args.args[0].arg
  if isinstance(e,ast.BinOp) and isinstance(e.op,ast.Pow) and is_x(e.left) and isinstance(e.right,ast.Constant) and e.right.value==3:return True
  if isinstance(e,ast.BinOp) and isinstance(e.op,ast.Mult):
   if is_x(e.left) and isinstance(e.right,ast.BinOp) and isinstance(e.right.op,ast.Mult) and is_x(e.right.left) and is_x(e.right.right):return True
   if is_x(e.right) and isinstance(e.left,ast.BinOp) and isinstance(e.left.op,ast.Mult) and is_x(e.left.left) and is_x(e.left.right):return True
  return False
 except SyntaxError:return False
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-dir",type=Path,default=Path("hf_model"))
 p.add_argument("--checkpoint",type=Path,default=Path("stage6_tiny/checkpoints/best.pt"))
 p.add_argument("--output",type=Path,default=Path("stage6_tiny/unseen_18b.json"))
 p.add_argument("--max-new",type=int,default=48)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 if not torch.cuda.is_available():p.error("CUDA required")
 paths=[a.checkpoint,a.model_dir/"tokenizer.model",a.model_dir/"model.safetensors"]
 before={str(x):sha(x) for x in paths}
 sp=spm.SentencePieceProcessor(model_file=str(a.model_dir/"tokenizer.model"))
 sys.path.insert(0,str(a.model_dir.resolve()))
 from inference import load_svarog
 model,_=load_svarog(a.model_dir.resolve(),device="cpu")
 state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
 model.load_state_dict(state["model"],strict=True)
 del state
 model=model.to("cuda").eval()
 results=[]
 for category,q,keys in TESTS:
  answer=infer(model,sp,SVAROG_SYSTEM,q,a.max_new)
  passed=code_check(answer) if category=="code" else any(k in answer.casefold() for k in keys)
  results.append({"category":category,"question":q,"answer":answer,"pass":passed})
  print(("PASS" if passed else "FAIL"),category,q,"=>",repr(answer),flush=True)
 after={str(x):sha(x) for x in paths}
 if before!=after:raise RuntimeError("Checkpoint or model inputs changed")
 report={"status":"COMPLETE","purpose":"unseen prompt probe, not benchmark","checkpoint_sha256":before[str(a.checkpoint)],"score":sum(x["pass"] for x in results),"total":len(results),"results":results,"inputs_unchanged":True}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18B UNSEEN",report["score"],"/",report["total"],flush=True)
if __name__=="__main__":main()
