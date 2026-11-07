#!/usr/bin/env python3
"""Stage17A read-only corpus audit: exact SFT masks, lengths, repetition and source mix."""
import argparse, collections, json, random, re, statistics
from pathlib import Path
import sentencepiece as spm
from stage4_train_sft import encode_record

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=Path("data_stage3_sft"))
    p.add_argument("--tokenizer",type=Path,default=Path("hf_model/tokenizer.model"))
    p.add_argument("--output",type=Path,default=Path("stage4/audit/stage17a.json"))
    p.add_argument("--seq",type=int,default=512)
    p.add_argument("--examples",type=int,default=5)
    a=p.parse_args()
    if a.output.exists():p.error("Refusing to overwrite audit")
    sp=spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    summary={}
    for split in ("train","validation"):
        source=collections.defaultdict(lambda:collections.Counter())
        lengths=collections.defaultdict(list)
        samples=collections.defaultdict(list)
        topic=collections.Counter()
        seen=set()
        issues=collections.Counter()
        rng=random.Random(17)
        with (a.data_dir/(split+".jsonl")).open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                row=json.loads(line)
                origin=row.get("origin","unknown")
                msgs=row["messages"]
                q,ans=msgs[-2]["content"],msgs[-1]["content"]
                key=(q.strip().casefold(),ans.strip().casefold())
                if key in seen:issues["duplicate_pair"]+=1
                seen.add(key)
                try:enc=encode_record(row,sp,a.seq)
                except (ValueError,KeyError,TypeError) as e:
                    issues["invalid_record"]+=1
                    continue
                prefix="".join(f"[im_start]{m['role']}\n{m['content'].strip()}[im_end]\n" for m in msgs[:-1])+"[im_start]assistant\n"
                plen=1+len(sp.encode(prefix))
                alen=len(sp.encode(ans.strip()+"[im_end]\n"))+1
                lengths[origin].append((plen,alen))
                source[origin]["total"]+=1
                if enc is None:
                    source[origin]["too_long"]+=1
                else:
                    ids,labels=enc
                    if len(ids)!=len(labels) or any(x!=-100 for x in labels[:plen]) or labels[plen:]!=ids[plen:]:
                        issues["mask_mismatch"]+=1
                    source[origin]["kept"]+=1
                if len(ans.split())>20:
                    words=re.findall(r"\w+",ans.lower())
                    if words:
                        top=collections.Counter(words).most_common(1)[0][1]
                        if top/len(words)>.25:
                            issues["high_word_repetition"]+=1
                low=q.lower()
                category=("code" if any(t in low for t in ("python","код","функци","script","function","javascript"))
                          else "math" if re.search(r"\d+\\s*[+*/=-]\\s*\d+",low) else
                          "explanation" if any(t in low for t in ("объясни","explain","почему","why","what is")) else
                          "dialogue" if any(t in low for t in ("привет","hello","hi ","как дела")) else "other")
                topic[category]+=1
                if len(samples[origin])<a.examples:
                    samples[origin].append({"question":q[:300],"answer":ans[:500],"prefix_tokens":plen,"answer_tokens":alen,"fits":enc is not None})
        def pct(v,p):
            return sorted(v)[min(len(v)-1,int((len(v)-1)*p))] if v else None
        summary[split]={"sources":{k:dict(v) for k,v in source.items()},
                        "token_lengths":{k:{"prefix_p50":pct([x for x,y in v],.5),
                            "prefix_p95":pct([x for x,y in v],.95),
                            "answer_p50":pct([y for x,y in v],.5),
                            "answer_p95":pct([y for x,y in v],.95)} for k,v in lengths.items()},
                        "topics_heuristic":dict(topic),"issues":dict(issues),"examples":dict(samples)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("STAGE17A AUDIT",a.output)
    for split,info in summary.items():
        print(split,"sources",info["sources"],"issues",info["issues"],"topics",info["topics_heuristic"])
        print("LENGTHS",info["token_lengths"])
    print("WARNING: topic labels are keyword heuristics; inspect sampled examples in JSON.")
if __name__=="__main__":main()
