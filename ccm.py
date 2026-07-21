import json
import random
import time
import traceback
from collections import defaultdict
import zstandard as zstd
from jinja2 import Template
from pathlib import Path

class CCM():
    def __init__(self,dataset=None,file=None,format="",zip=True,add_prefix=True,gram_count=5,repeat=False,save=True,to_set=True,hipo_model={}, hipo=[],hipo_multiple=1):

        self.repeat=repeat
        cctx = zstd.ZstdCompressor(level=22)
        dctx = zstd.ZstdDecompressor()
        self.hipo=hipo
        self.hipo_multiple=hipo_multiple
        if dataset!=None:
            self.dataset=dataset
            self.wordset={}
            self.name=""
            self.gram_count=gram_count
            self.format=format
            s_time = time.time()
            print("Indexing...")
            prefix = self.indexing(self.dataset)
            suffix = f"{prefix // 1_000_000_000}B" if prefix >= 1_000_000_000 else f"{prefix // 1_000_000}M" if prefix >= 1_000_000 else f"{prefix // 1000}K" if prefix > 1000 else f"{prefix}"
            prefix = f"_{suffix}"
            if add_prefix:
                file+=prefix
            print(prefix)
            print("Time:", time.time() - s_time)
            print("Save...")
            original_text = json.dumps({"format": self.format, "gram_count": self.gram_count, "wordset": self.wordset})
            self.name=file
            self.model_name = Path(file).stem
            if zip:
                if save:
                    compressed_data = cctx.compress(original_text.encode('utf-8'))
                    with open(f"{file}.cws","wb") as f:
                        f.write(compressed_data)

            else:
                with open(f"{file}.json","w",encoding="utf-8") as f:
                    f.write(original_text)

        else:
            with open(file,"rb") as f:
                if file.lower().endswith('.cws'):
                    readed = json.loads(dctx.decompress(f.read()).decode('utf-8'))
                else:
                    readed =json.loads(f.read())
                self.model_name = Path(file).stem
                self.wordset = readed["wordset"]
                self.to_set=to_set
                self.gram_count = readed["gram_count"]
                self.format = readed["format"]
                if readed.get("hipo"):
                    self.hipo =  set([frozenset(context) for context in readed["hipo"]])
                if readed.get("hipo_model"):
                    self.hipo_model=readed["hipo_model"]
                if readed.get("hipo_multiple"):
                    self.hipo_multiple = readed["hipo_multiple"]
                if hipo_model!={} and True:
                    self.wordset = self.merge_dicts(self.wordset,self.hipo_model)
                if to_set:
                    for word in self.wordset:
                        self.wordset[word] = [set(context) for context in self.wordset[word]]

    def chat(self,messages,max_tokens=250,end_tokens=[],ignore_tokens=[],load_function=None,print_debug=False,temperature=0,more_info=False):
        s_time = time.time()
        template = Template(self.format)
        formatted = template.render(messages=messages).lower()
        to_ret=self.generate(formatted,max_tokens,end_tokens,ignore_tokens,load_function,print_debug,temperature)
        #print("Time:", time.time() - s_time)
        if not more_info:return to_ret
        else: return to_ret,formatted+to_ret

    def merge_dicts(self,d1, d2):
        result = defaultdict(list)
        for d in (d1, d2):
            for k, v in d.items():
                if isinstance(v, list):
                    result[k].extend(v)
                else:
                    result[k] = v
        return dict(result)

    def generate(self,text_user,max_tokens=250,end_tokens=[],ignore_tokens=[],load_function=None,print_debug=False,temperature=0):
        text=text_user
        if load_function!=None:
            inter_result = load_function(text_user)
            self.wordset = self.merge_dicts(self.wordset, self.indexing(inter_result,train=False))

        new_text=""
        for i in range(max_tokens):
            if not text or len(text) < self.gram_count:
                q_trigrams = {text} if text else set()
            else:
                q_trigrams = set(text[i:i + self.gram_count] for i in range(len(text) - 2))
            len_q = len(q_trigrams)
            word_win = ["", -999]
            last_winers=["","",""]
            if print_debug: print()
            try:
                for num_word,word in enumerate(self.wordset):
                    for data in self.wordset[word]:
                            try:

                                intersection = len(q_trigrams & data)
                                union_len = len_q + len(data) - intersection
                                score = intersection / union_len if union_len else 0.0

                                if data in self.hipo:
                                   score*=self.hipo_multiple
                                res = [[text, score]]
                                if temperature!=0: res[0][1] *= (random.uniform(-temperature, temperature))+1
                                if word in ignore_tokens:
                                    res[0][1] -= 0.2
                                if word_win[1] < res[0][1]:
                                    if print_debug: print("[", word, res[0][1], "]")
                                    word_win = [word, res[0][1]]
                                    last_winers.insert(0, word_win[0])
                                    last_winers.pop()
                                if print_debug: print(f"\r{num_word}/{len(self.wordset)}", end="",flush=True)
                            except ValueError as e:
                                continue
            except Exception as e:
                if print_debug: traceback.print_exc()

            if not word_win[0] in new_text or self.repeat:
                for token in end_tokens:
                    if token in word_win[0]:
                        if print_debug: print()
                        return new_text.strip()
                text += f" {word_win[0]}"
                new_text += f" {word_win[0]}"
                if print_debug: print(" score: "+str(word_win[1]), end="")
                if print_debug: print(new_text, end=" ")
            else:
                if print_debug: print()
                return new_text.strip()
        if print_debug: print()
        return new_text.strip()


    def indexing(self,dataset,train=True):
        edited_words = sorted(list(set(" ".join(dataset).split())))
        words_in_dataset = {}
        for num_word, word in enumerate(edited_words):
            for data in dataset:
                if word in data:
                    try:
                        data = " ".join(data.split()[:data.split().index(word)])
                        if words_in_dataset.get(word) == None:
                            words_in_dataset[word] = []
                        if len(data) < self.gram_count:
                            t_trigrams = {data} if data else set()
                        else:
                            t_trigrams = set(data[i:i + self.gram_count] for i in range(len(data) - 2))
                        words_in_dataset[word].append(list(t_trigrams))
                        if train: print(f"\r{num_word}/{len(edited_words)}", end="", flush=True)
                    except ValueError as e:
                        continue
        if train:
            self.wordset=words_in_dataset
            return sum(len(group) for contexts in self.wordset.values() for group in contexts)
        else:
            return words_in_dataset
