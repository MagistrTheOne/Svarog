#!/usr/bin/env python3
"""Stage18L dataset: canonical disjoint numeric tasks and held-out wording.
Only TRAIN tasks are used for training/mining. Validation is not trained on.
The old Stage18K test is never read. This is a development experiment, not a new sealed test.
"""
import argparse,collections,hashlib,json,random
from pathlib import Path
from stage3_sft_persona import SVAROG_SYSTEM

TRAIN_TEMPLATES={
 "successor":["Какое число идёт после {a}?","Что идёт сразу за числом {a}?","Продолжи счёт: {a}, ...","Прибавь единицу к {a}.","Укажи следующее целое число после {a}."],
 "predecessor":["Какое число идёт перед {a}?","Что находится непосредственно перед {a}?","Назови число на единицу меньше {a}.","Отними единицу от {a}.","Продолжи назад: {a}, ..."],
 "compare":["Какое число больше: {a} или {b}?","Выбери большее из {a} и {b}.","Что больше, {a} или {b}?","Укажи максимум из двух чисел: {a}, {b}.","Сравни {a} и {b}; ответь большим числом."]
}
HELD_TEMPLATES={
 "successor":["Назови число, следующее за {a}.","Какое целое число непосредственно следует за {a}?"],
 "predecessor":["Назови число, предшествующее {a}.","Какое целое число непосредственно предшествует {a}?"],
 "compare":["Из двух чисел {a} и {b} назови большее.","Какое из чисел {a} и {b} имеет большее значение?"]
}
def split(key):
 h=int.from_bytes(hashlib.sha256(("stage18L-v1|"+repr(key)).encode()).digest()[:8],"big")%100
 return "train" if h<80 else "validation" if h<90 else "holdout"
def record(cat,key,question,answer,template_id):
 return {"category":cat,"canonical_key":list(key),"template_id":template_id,
         "messages":[{"role":"system","content":SVAROG_SYSTEM},
                     {"role":"user","content":question},
                     {"role":"assistant","content":str(answer)+"."}]}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18l_concepts"))
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists")
 rng=random.Random(1812)
 sets={s:[] for s in ("train","validation","holdout","validation_paraphrase")}
 keys=collections.defaultdict(set)
 tasks=[]
 for n in range(0,101):
  tasks.append(("successor",("successor",n),n,None,n+1))
 for n in range(1,102):
  tasks.append(("predecessor",("predecessor",n),n,None,n-1))
 for x in range(0,40):
  for y in range(x+1,40):
   tasks.append(("compare",("compare",x,y),x,y,y))
 for cat,key,a1,b1,answer in tasks:
  s=split(key);keys[s].add(key)
  if s=="holdout":continue # reserved, never loaded by trainer
  orientations=[(a1,b1)] if cat!="compare" else [(a1,b1),(b1,a1)]
  if s=="train":
   for a2,b2 in orientations:
    for i,t in enumerate(TRAIN_TEMPLATES[cat]):
     sets["train"].append(record(cat,key,t.format(a=a2,b=b2),answer,f"train_{i}"))
  else:
   for a2,b2 in orientations:
    for i,t in enumerate(TRAIN_TEMPLATES[cat]):
     sets["validation"].append(record(cat,key,t.format(a=a2,b=b2),answer,f"train_{i}"))
    for i,t in enumerate(HELD_TEMPLATES[cat]):
     sets["validation_paraphrase"].append(record(cat,key,t.format(a=a2,b=b2),answer,f"held_{i}"))
 assert all(keys[x].isdisjoint(keys[y]) for x,y in (("train","validation"),("train","holdout"),("validation","holdout")))
 a.output.mkdir(parents=True)
 for s,rows in sets.items():
  rng.shuffle(rows)
  with (a.output/(s+".jsonl")).open("w",encoding="utf-8") as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
  print(s,len(rows),dict(collections.Counter(r["category"] for r in rows)),flush=True)
 (a.output/"manifest.json").write_text(json.dumps({"seed":1812,"split":"canonical task 80/10/10, deterministic SHA256","heldout":"10% reserved but NOT materialized","validation_paraphrase":"unseen templates AND unseen canonical numeric tasks","warning":"Stage18K sealed test is not read. Development validation is not independent test."},ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18L DATA READY",flush=True)
if __name__=="__main__":main()
