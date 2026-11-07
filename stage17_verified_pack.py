#!/usr/bin/env python3
"""Stage17C.1: verified arithmetic and reviewed micro-instructions, no network or checkpoint writes."""
import argparse, collections, hashlib, json, random, shutil
from pathlib import Path
import sentencepiece as spm
from stage3_sft_persona import SVAROG_SYSTEM
from stage4_train_sft import encode_record

FACTS=[
 ("Столица Франции?","Париж.","What is the capital of France?","Paris."),
 ("Столица Германии?","Берлин.","What is the capital of Germany?","Berlin."),
 ("Столица Италии?","Рим.","What is the capital of Italy?","Rome."),
 ("Столица Японии?","Токио.","What is the capital of Japan?","Tokyo."),
 ("Сколько дней в неделе?","Семь.","How many days are in a week?","Seven."),
 ("Сколько минут в часе?","60.","How many minutes are in an hour?","60."),
 ("Какой химический символ кислорода?","O.","What is the chemical symbol for oxygen?","O."),
 ("Сколько байтов в килобайте по определению SI?","1000.","How many bytes are in a kilobyte under SI?","1000."),
]
CODE=[
 ("Напиши Python-функцию double(x).","def double(x):\n    return x * 2",
  "Write a Python function double(x).","def double(x):\n    return x * 2"),
 ("Напиши Python-функцию is_even(n).","def is_even(n):\n    return n % 2 == 0",
  "Write a Python function is_even(n).","def is_even(n):\n    return n % 2 == 0"),
 ("Напиши Python-функцию absolute(x).","def absolute(x):\n    return abs(x)",
  "Write a Python function absolute(x).","def absolute(x):\n    return abs(x)"),
 ("Напиши Python-функцию add(a, b).","def add(a, b):\n    return a + b",
  "Write a Python function add(a, b).","def add(a, b):\n    return a + b"),
]
UNCERTAINTY=[
 ("Какой у меня пароль от роутера?","Я не знаю ваш пароль и не имею доступа к роутеру.",
  "What is my router password?","I do not know your password or have access to your router."),
 ("Назови серийный номер моего монитора.","У меня нет данных о серийном номере вашего монитора.",
  "Tell me my monitor's serial number.","I do not have your monitor's serial number."),
 ("Какая температура на улице прямо сейчас?","Без актуальных данных о погоде и местоположении не могу назвать температуру.",
  "What is the temperature outside right now?","I need current weather data and your location to answer."),
 ("Какой результат матча, который идёт сейчас?","Нужны название матча и актуальные результаты; выдумывать счёт не буду.",
  "What is the score of the game happening now?","I need the match and live results; I will not invent a score."),
]
EXPLAIN=[
 ("Что такое токен в языковой модели?","Токен — единица текста, которую обрабатывает модель: слово, часть слова или знак.",
  "What is a token in a language model?","A token is a unit of text processed by a model, such as a word, word part, or symbol."),
 ("Что такое переобучение модели?","Переобучение — когда модель хорошо запоминает обучающие примеры, но плохо работает на новых.",
  "What is overfitting?","Overfitting is when a model learns training examples too closely and performs poorly on new data."),
]
def row(q,a,lang,kind):
    return {"messages":[{"role":"system","content":SVAROG_SYSTEM},
                        {"role":"user","content":q},{"role":"assistant","content":a}],
            "origin":"NULLXES_verified_v1","language":lang,"persona":False,"stage17_category_heuristic":kind}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,default=Path("data_stage5_sft"))
    p.add_argument("--output-dir",type=Path,default=Path("data_stage5_verified"))
    p.add_argument("--tokenizer",type=Path,default=Path("hf_model/tokenizer.model"))
    p.add_argument("--seq",type=int,default=512)
    p.add_argument("--math-train",type=int,default=1200)
    p.add_argument("--math-validation",type=int,default=120)
    p.add_argument("--seed",type=int,default=17171)
    a=p.parse_args()
    if a.output_dir.exists():p.error("Output exists; refusing overwrite")
    if min(a.math_train,a.math_validation)<1:p.error("Math counts must be positive")
    sp=spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    rng=random.Random(a.seed)
    train=[]; val=[]
    # Train/validation arithmetic are disjoint in both input numbers and prompt patterns.
    train_patterns={
      "ru":["Вычисли: {x} {op} {y}.","Сколько будет {x} {op} {y}?","Реши пример {x} {op} {y}."],
      "en":["Calculate {x} {op} {y}.","What is {x} {op} {y}?","Solve {x} {op} {y}."]}
    val_patterns={"ru":["Найди результат выражения {x} {op} {y}."],
                  "en":["Give the result of {x} {op} {y}."]}
    for target,count,lo,hi,patterns in [
        (train,a.math_train,1,70,train_patterns),
        (val,a.math_validation,80,130,val_patterns)]:
        candidates=[]
        for x in range(lo,hi+1):
            for y in range(lo,hi+1):
                for op in ("+","-","*"):
                    z={"+":x+y,"-":x-y,"*":x*y}[op]
                    for lang in ("ru","en"):
                        candidates.append((x,y,op,z,lang))
        rng.shuffle(candidates)
        for x,y,op,z,lang in candidates:
            template=rng.choice(patterns[lang])
            question=template.format(x=x,y=y,op=op)
            answer=str(z)
            target.append(row(question,answer,lang,"math_verified"))
            if len(target)>=count:break
        if len(target)!=count:raise RuntimeError("Not enough arithmetic examples")
    # Human-reviewed fixed micro examples; put in train only, not in held-out validation.
    for kind,group in [("facts",FACTS),("code",CODE),("uncertainty",UNCERTAINTY),("explanation",EXPLAIN)]:
        for ruq,rua,enq,ena in group:
            train.extend((row(ruq,rua,"ru",kind),row(enq,ena,"en",kind)))
    # Deduplicate across original and verified datasets; keep original held-out split separate.
    existing={}
    seen=set()
    def key(item):
        m=item["messages"]
        return hashlib.sha256((m[1]["content"].strip().casefold()+"\n"+m[2]["content"].strip().casefold()).encode()).hexdigest()
    for split in ("train","validation"):
        original=[]
        with (a.input_dir/(split+".jsonl")).open(encoding="utf-8") as f:
            for line in f:
                item=json.loads(line); k=key(item)
                if k in seen:continue
                seen.add(k);original.append(item)
        existing[split]=original
    additions={"train":[],"validation":[]}
    for split,items in (("train",train),("validation",val)):
        for item in items:
            k=key(item)
            if k in seen:continue
            if encode_record(item,sp,a.seq) is None:
                raise RuntimeError("Verified example does not fit seq")
            seen.add(k);additions[split].append(item)
    if not additions["train"] or not additions["validation"]:
        raise RuntimeError("No new verified examples")
    a.output_dir.mkdir(parents=True)
    counts={}
    for split in ("train","validation"):
        all_rows=existing[split]+additions[split]
        rng.shuffle(all_rows)
        with (a.output_dir/(split+".jsonl")).open("w",encoding="utf-8") as f:
            for item in all_rows:f.write(json.dumps(item,ensure_ascii=False)+"\n")
        counts[split]={"original":len(existing[split]),"verified_added":len(additions[split]),"total":len(all_rows)}
    manifest={"counts":counts,"seed":a.seed,"seq":a.seq,"verified_origin":"NULLXES_verified_v1",
              "verification":"Arithmetic computed in Python; fixed facts and code manually authored, not externally sourced",
              "warning":"Arithmetic training and validation use disjoint number ranges and templates; other original dataset examples remain heuristic and may be noisy."}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("STAGE17C.1 VERIFIED PACK READY",counts,flush=True)
    print("No original files modified; inspect manifest and samples before training.",flush=True)
if __name__=="__main__":main()
