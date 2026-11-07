#!/usr/bin/env python3
"""Stage17C: build smaller auditable SFT from existing corpus; no network or synthetic padding."""
import argparse,collections,hashlib,json,random,re
from pathlib import Path
import sentencepiece as spm
from stage4_train_sft import encode_record

def category(q):
    s=q.lower()
    if any(x in s for x in ("python","javascript","напиши код","функцию","function","write code")):return "code"
    if re.search(r"\d+\s*[+\-*/]\s*\d+",s) or any(x in s for x in ("сколько будет","calculate","арифметик")):return "math"
    if any(x in s for x in ("объясни","explain","what is","что такое","почему")):return "explanation"
    if any(x in s for x in ("пароль","password","serial number","не знаешь","неизвестно")):return "uncertainty"
    if any(x in s for x in ("привет","hello","как дела","hi there")):return "dialogue"
    return "general"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,default=Path("data_stage3_sft"))
    p.add_argument("--output-dir",type=Path,default=Path("data_stage5_sft"))
    p.add_argument("--tokenizer",type=Path,default=Path("hf_model/tokenizer.model"))
    p.add_argument("--seq",type=int,default=512)
    p.add_argument("--max-answer-tokens",type=int,default=160)
    p.add_argument("--max-per-source-language",type=int,default=8000)
    p.add_argument("--seed",type=int,default=1701)
    a=p.parse_args()
    if a.output_dir.exists():p.error("Output directory exists; refusing overwrite")
    sp=spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    rng=random.Random(a.seed)
    outputs={}
    stats=collections.Counter()
    for split in ("train","validation"):
        groups=collections.defaultdict(list)
        seen=set()
        with (a.input_dir/(split+".jsonl")).open(encoding="utf-8") as f:
            for line in f:
                row=json.loads(line)
                messages=row["messages"]
                question=messages[1]["content"].strip()
                answer=messages[2]["content"].strip()
                if not question or not answer:continue
                encoded=encode_record(row,sp,a.seq)
                if encoded is None:
                    stats["too_long"]+=1;continue
                ids,labels=encoded
                answer_tokens=sum(v!=-100 for v in labels)
                if answer_tokens>a.max_answer_tokens:
                    stats["long_answer"]+=1;continue
                if answer_tokens<5:
                    stats["short_answer"]+=1;continue
                words=re.findall(r"\w+",answer.casefold())
                if len(words)>12 and collections.Counter(words).most_common(1)[0][1]/len(words)>.20:
                    stats["repetitive"]+=1;continue
                key=hashlib.sha256((question.casefold()+"\n"+answer.casefold()).encode()).hexdigest()
                if key in seen:stats["duplicate"]+=1;continue
                seen.add(key)
                cat=category(question)
                row["stage17_category_heuristic"]=cat
                groups[(row.get("origin","unknown"),row.get("language","unknown"))].append(row)
        selected=[]
        for key,items in sorted(groups.items()):
            rng.shuffle(items)
            take=items[:a.max_per_source_language] if split=="train" else items
            selected.extend(take)
            stats[f"{split}:{key[0]}:{key[1]}"]=len(take)
        rng.shuffle(selected)
        outputs[split]=selected
    if not outputs["train"] or not outputs["validation"]:
        p.error("Empty curated split")
    a.output_dir.mkdir(parents=True)
    for split,rows in outputs.items():
        with (a.output_dir/(split+".jsonl")).open("w",encoding="utf-8") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifest={"counts":{"train":len(outputs["train"]),"validation":len(outputs["validation"])},
              "stats":dict(stats),"source":"Stage14 existing SFT; no new downloads","seq":a.seq,
              "max_answer_tokens":a.max_answer_tokens,"seed":a.seed,
              "warning":"Heuristic cleaning only. Human factuality/license review required; no guarantee of correct answers."}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("STAGE17C BUILT",manifest["counts"],"at",a.output_dir)
    print("SOURCE COUNTS",dict(stats))
if __name__=="__main__":main()
