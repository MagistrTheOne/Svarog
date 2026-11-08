#!/usr/bin/env python3
"""Stage18A: construct 32 short, known-answer train examples; validation intentionally equals train for memorization only."""
import argparse,json,random,shutil
from pathlib import Path
import sentencepiece as spm
from stage3_sft_persona import SVAROG_SYSTEM
from stage4_train_sft import encode_record

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,default=Path("data_stage18_tiny"))
 p.add_argument("--tokenizer",type=Path,default=Path("hf_model/tokenizer.model"))
 p.add_argument("--seq",type=int,default=512)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; refusing overwrite")
 sp=spm.SentencePieceProcessor(model_file=str(a.tokenizer))
 samples=[]
 pairs=[
 ("Кто ты?","Я Сварог, языковая модель NULLXES."),
 ("Какая столица Франции?","Париж."),
 ("Какая столица Германии?","Берлин."),
 ("Какая столица Италии?","Рим."),
 ("Какая столица Японии?","Токио."),
 ("Сколько будет 2 + 2?","4."),
 ("Сколько будет 7 - 3?","4."),
 ("Сколько будет 7 - 3 + 2?","6."),
 ("Сколько будет 6 * 3?","18."),
 ("Сколько будет 12 + 8?","20."),
 ("Сколько будет 9 - 4?","5."),
 ("Сколько будет 5 * 5?","25."),
 ("Сколько минут в часе?","60."),
 ("Сколько дней в неделе?","7."),
 ("Назови символ кислорода.","O."),
 ("Назови символ водорода.","H."),
 ("Напиши Python функцию square(x).","def square(x):\n    return x * x"),
 ("Напиши Python функцию add(a, b).","def add(a, b):\n    return a + b"),
 ("Напиши Python функцию double(x).","def double(x):\n    return x * 2"),
 ("Напиши Python функцию is_even(n).","def is_even(n):\n    return n % 2 == 0"),
 ("Какой у меня пароль?","Я не знаю ваш пароль."),
 ("Какой серийный номер моего ноутбука?","У меня нет этих данных."),
 ("Ты человек?","Нет, я языковая модель."),
 ("Ты знаешь всё?","Нет, я могу ошибаться."),
 ("Что такое токен?","Токен — единица текста для языковой модели."),
 ("What is the capital of France?","Paris."),
 ("What is the capital of Germany?","Berlin."),
 ("What is 2 + 2?","4."),
 ("What is 7 - 3 + 2?","6."),
 ("Write Python square(x).","def square(x):\n    return x * x"),
 ("What is your name?","I am SVAROG, a NULLXES language model."),
 ("Do you know my password?","No, I do not know your password.")
 ]
 for q,answer in pairs:
  item={"origin":"NULLXES_tiny_diagnostic","messages":[{"role":"system","content":SVAROG_SYSTEM},{"role":"user","content":q},{"role":"assistant","content":answer}]}
  if encode_record(item,sp,a.seq) is None:raise RuntimeError("Too long: "+q)
  samples.append(item)
 a.output.mkdir(parents=True)
 for split in ("train","validation"):
  with (a.output/(split+".jsonl")).open("w",encoding="utf-8") as f:
   for row in samples:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 (a.output/"manifest.json").write_text(json.dumps({"purpose":"memorization diagnostic ONLY","count":len(samples),"validation_equals_train":True,"not_for_generalization":True},indent=2),encoding="utf-8")
 print("STAGE18 TINY READY",len(samples),"examples; train=validation by design; NEVER report as generalization")
if __name__=="__main__":main()
