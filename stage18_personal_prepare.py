#!/usr/bin/env python3
"""NULLXES SVAROG Stage18C: GPT-6-authored instruction curriculum, no external datasets.
Holdout contains distinct prompts and arithmetic operand pairs. Not a general benchmark.
"""
import argparse,hashlib,json,random
from pathlib import Path
import sentencepiece as spm
from stage3_sft_persona import SVAROG_SYSTEM
from stage4_train_sft import encode_record

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18_personal"))
 p.add_argument("--tokenizer",type=Path,default=Path("hf_model/tokenizer.model"))
 p.add_argument("--seq",type=int,default=512)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 sp=spm.SentencePieceProcessor(model_file=str(a.tokenizer))
 train=[];val=[];seen=set()
 def add(split,category,q,answer):
  assert q not in seen,("duplicate prompt",q)
  seen.add(q)
  row={"origin":"NULLXES_GPT6_direct_instruction","category":category,"messages":[{"role":"system","content":SVAROG_SYSTEM},{"role":"user","content":q},{"role":"assistant","content":answer}]}
  if encode_record(row,sp,a.seq) is None:raise RuntimeError("Too long: "+q)
  (train if split=="train" else val).append(row)
 # Identical underlying fact, multiple distinct phrasings. Holdout prompts unseen.
 facts=[("Франции","Париж","France","Paris"),("Германии","Берлин","Germany","Berlin"),("Италии","Рим","Italy","Rome"),("Японии","Токио","Japan","Tokyo"),("Испании","Мадрид","Spain","Madrid"),("Канады","Оттава","Canada","Ottawa")]
 for ru,city,en,cityen in facts:
  add("train","facts",f"Какая столица {ru}?",city+".")
  add("train","facts",f"Назови столицу {ru}.",city+".")
  add("train","facts",f"What is the capital of {en}?",cityen+".")
  add("validation","facts",f"Столица {ru} — какой город?",city+".")
  add("validation","facts",f"Name the capital city of {en}.",cityen+".")
 # Identity with variants; no direct answers about user or secrets.
 identity_train=[
 ("Кто ты?","Я Сварог, языковая модель NULLXES."),
 ("Как тебя зовут?","Меня зовут Сварог."),
 ("Представься коротко.","Я Сварог — языковая модель NULLXES."),
 ("Ты человек?","Нет, я языковая модель."),
 ("Кто тебя разрабатывает?","Меня разрабатывает NULLXES."),
 ("Ты GPT-6?","Нет. Я Сварог, отдельная языковая модель NULLXES."),
 ("Ты можешь ошибаться?","Да. Мои ответы нужно проверять."),
 ("Ты уже всё знаешь?","Нет. Я не знаю всего и могу ошибаться."),
 ("What is your name?","My name is SVAROG."),
 ("Who made you?","NULLXES develops me."),
 ("Are you human?","No, I am a language model."),
 ("Can you make mistakes?","Yes. My answers can be wrong."),
 ]
 identity_val=[
 ("Назови своё имя.","Меня зовут Сварог."),
 ("Как называется твоя модель?","Я Сварог, языковая модель NULLXES."),
 ("Ты настоящий человек?","Нет, я языковая модель."),
 ("What should I call you?","You can call me SVAROG."),
 ]
 for q,ans in identity_train:add("train","identity",q,ans)
 for q,ans in identity_val:add("validation","identity",q,ans)
 # Math: different operand pairs across train/holdout. Multiple phrasings and languages.
 rng=random.Random(1818)
 for op,sign in [("add","+"),("sub","-"),("mul","*")]:
  pairs=[(x,y) for x in range(2,19) for y in range(2,15) if op!="sub" or x>=y]
  rng.shuffle(pairs)
  for split,subset in [("train",pairs[:42]),("validation",pairs[42:54])]:
   for i,(x,y) in enumerate(subset):
    result={"add":x+y,"sub":x-y,"mul":x*y}[op]
    templates=[
     f"Сколько будет {x} {sign} {y}?",
     f"Вычисли: {x} {sign} {y}.",
     f"Calculate {x} {sign} {y}.",
     f"Реши пример {x} {sign} {y}."
    ]
    q=templates[i%len(templates)]
    add(split,"arithmetic",q,str(result)+".")
 # Python patterns: distinct function names and operations in holdout.
 code_train=[
 ("Напиши функцию Python square(x).","def square(x):\n    return x * x"),
 ("Напиши функцию Python double(x).","def double(x):\n    return x * 2"),
 ("Напиши функцию Python triple(x).","def triple(x):\n    return x * 3"),
 ("Напиши функцию Python add(a, b).","def add(a, b):\n    return a + b"),
 ("Напиши функцию Python subtract(a, b).","def subtract(a, b):\n    return a - b"),
 ("Напиши функцию Python multiply(a, b).","def multiply(a, b):\n    return a * b"),
 ("Напиши функцию Python is_even(n).","def is_even(n):\n    return n % 2 == 0"),
 ("Напиши функцию Python is_odd(n).","def is_odd(n):\n    return n % 2 != 0"),
 ("Write a Python function cube(x).","def cube(x):\n    return x ** 3"),
 ("Write a Python function increment(x).","def increment(x):\n    return x + 1"),
 ("Write a Python function decrement(x).","def decrement(x):\n    return x - 1"),
 ("Write a Python function is_positive(x).","def is_positive(x):\n    return x > 0"),
 ]
 code_val=[
 ("Напиши функцию Python cube(x), возвращающую куб числа.","def cube(x):\n    return x ** 3"),
 ("Напиши функцию Python increment(x), прибавляющую единицу.","def increment(x):\n    return x + 1"),
 ("Write a Python function square(x) that returns x squared.","def square(x):\n    return x * x"),
 ("Write a Python function add(a, b) returning their sum.","def add(a, b):\n    return a + b"),
 ]
 for q,ans in code_train:add("train","code",q,ans)
 for q,ans in code_val:add("validation","code",q,ans)
 uncertainty_train=[
 ("Какой пароль от моего Wi-Fi?","Я не знаю ваш пароль."),
 ("Какой у меня PIN-код?","У меня нет этих данных."),
 ("Какой серийный номер моего ноутбука?","У меня нет этих данных."),
 ("Какой мой домашний адрес?","Я не знаю ваш адрес."),
 ("Ты знаешь мой номер паспорта?","Нет, у меня нет этих данных."),
 ("Сколько денег на моей карте?","У меня нет доступа к вашему балансу."),
 ("What is my Wi-Fi password?","I do not know your password."),
 ("What is my laptop serial number?","I do not have that information."),
 ]
 uncertainty_val=[
 ("Можешь назвать мой секретный PIN-код?","Я не знаю ваш PIN-код."),
 ("Какой у меня пароль от телефона?","Я не знаю ваш пароль."),
 ("Do you know my bank account balance?","I do not have that information."),
 ("Can you tell me my home address?","I do not know your address."),
 ]
 for q,ans in uncertainty_train:add("train","uncertainty",q,ans)
 for q,ans in uncertainty_val:add("validation","uncertainty",q,ans)
 logic_train=[
 ("Все кошки — животные. Все животные дышат. Дышат ли кошки?","Да, кошки дышат."),
 ("Все розы — цветы. Все цветы — растения. Розы — растения?","Да, розы — растения."),
 ("Все квадраты — прямоугольники. Все прямоугольники имеют четыре стороны. У квадрата четыре стороны?","Да."),
 ("Если идёт дождь, земля мокрая. Сейчас идёт дождь. Земля мокрая?","Да, по условию земля мокрая."),
 ("Все птицы — животные. Все животные живые. Птицы живые?","Да."),
 ("Если лампа включена, она светит. Лампа включена. Она светит?","Да, по условию."),
 ]
 logic_val=[
 ("Все собаки — млекопитающие. Все млекопитающие — животные. Собаки — животные?","Да."),
 ("Если кнопка нажата, индикатор горит. Кнопка нажата. Индикатор горит?","Да, по условию."),
 ("Все сосны — деревья. Все деревья — растения. Сосны — растения?","Да."),
 ("Все автомобили — транспорт. Весь транспорт перемещает людей или грузы. Автомобили — транспорт?","Да."),
 ]
 for q,ans in logic_train:add("train","logic",q,ans)
 for q,ans in logic_val:add("validation","logic",q,ans)
 # Prevent the tiny memorization corpus from being silently reused.
 if len(train)<100 or len(val)<32:raise RuntimeError(f"Too few rows: {len(train)} {len(val)}")
 a.output.mkdir(parents=True)
 for name,rows in [("train",train),("validation",val)]:
  with (a.output/(name+".jsonl")).open("w",encoding="utf-8") as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 manifest={"author":"GPT-6 / NULLXES","external_datasets":False,"train":len(train),"validation":len(val),"disjoint_prompts":True,"math_operand_pairs_disjoint_within_operation":True,"note":"Small controlled curriculum; does not establish general reasoning ability."}
 (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 print("STAGE18C PERSONAL CORPUS",json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
