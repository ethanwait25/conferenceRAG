print("Initializing...")

import numpy as np
import pandas as pd
import os
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI

NUM_CLUSTERS = 3
N_CLOSEST = 3

class ConferenceRAG():
    def __init__(self):
        self.strategies = {
            "fr-tlk": "free_embeddings/free_talks.csv",
            "fr-pg": "free_embeddings/free_paragraphs.csv",
            "fr-cl": f"free_embeddings/free_embeddings_{NUM_CLUSTERS}_clusters.csv",

            "oai-tlk": "openai_embeddings/openai_talks.csv",
            "oai-pg": "openai_embeddings/openai_paragraphs.csv",
            "oai-cl": f"openai_embeddings/openai_embeddings_{NUM_CLUSTERS}_clusters.csv"
        }

        self.init_openai()
        self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_embedding("oai-pg")
        self.prompt_mode = "q"

    def init_openai(self):
        with open("config.json") as config:
            openaiKey = json.load(config)["openaiKey"]
        OpenAI.api_key = openaiKey
        self.oai_client = OpenAI(api_key=OpenAI.api_key)
    
    def load_embedding(self, strategy):
        self.emb, self.emb_name = pd.read_csv(self.strategies[strategy], converters={'embedding': lambda s: np.fromstring(s.strip("[]"), sep=',', dtype=np.float32)}), strategy

    def handle_query(self, query):
        emb = self.get_query_embedding(query)
        closest = self.get_n_closest(emb, N_CLOSEST)
        return self.get_rag(query, closest)
    
    def get_n_closest(self, emb, n):
        X = np.stack(self.emb["embedding"].to_numpy()).astype(np.float32)
        emb = emb.astype(np.float32)

        dist = np.linalg.norm(X - emb, axis=1)
        topn = np.argpartition(dist, n)[:n]
        topn = topn[np.argsort(dist[topn])]

        nearest = self.emb.iloc[topn].copy()
        nearest["distance"] = dist[topn]
        nearest = nearest[["title", "speaker", "year", "text", "distance"]]
        return nearest
    
    def get_query_embedding(self, query):
        if self.emb_name.startswith("fr"):
            return self.transformer.encode(
                        query,
                        batch_size=32,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
        else:
            res = self.oai_client.embeddings.create(input=query, model="text-embedding-3-small")
            return np.array([item.embedding for item in res.data])

    def get_rag(self, query, closest):
        prompt_config = "rag_prompt_quote" if self.prompt_mode == "q" else "rag_prompt_explain"
        with open("config.json") as config:
            base_prompt = json.load(config)[prompt_config]
        
        prompt = []
        for _, row in closest.iterrows():
            prompt.append(row['text'])
            prompt.append(f"title: '{row['title']}'")
            prompt.append(f"speaker: '{row['speaker']}'")
            prompt.append(f"year: '{row['year']}'")
            prompt.append("")
        prompt.append(base_prompt)
        prompt.append(query)
        return self.getChatGptResponse("\n".join(prompt))

    def getChatGptResponse(self, message):
        stream = self.oai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}],
            stream=True
        )

        responses = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                responses.append(chunk.choices[0].delta.content)

        return "".join(responses)
    
    def loop(self):
        self.clear_console()
        print("Welcome to ConferenceRAG. What's on your mind?")
        print("(Use 'chmod' to update embedding/strategy mode. Use 'chres' to change response type.)")

        while True:
            try:
                line = input(f"{self.emb_name}:{self.prompt_mode} >>> ").lower()
                if line in ('q', 'quit', 'exit'):
                    break

                if line.split(" ")[0] == "chmod":
                    tokens = line.split(" ")
                    self.chmod(tokens[1] if len(tokens) > 1 else None)
                    continue

                if line.split(" ")[0] == "chres":
                    tokens = line.split(" ")
                    self.chres(tokens[1] if len(tokens) > 1 else None)
                    continue

                response = self.handle_query(line)
                print(response)

            except KeyboardInterrupt as k:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Till we meet at Jesus' feet.")
    
    def chmod(self, choice):
        if not choice or choice not in self.strategies:
            emb_ch = self.query_emb_ch()
            strat_ch = self.query_strat_ch()
            choice = f"{emb_ch}-{strat_ch}"
        print("Initializing, please wait...")
        self.load_embedding(choice)

    def query_emb_ch(self):
        print("'fr' = free | 'oai' = OpenAI")
        emb_ch = input(">>> Embedding: ").lower()
        while emb_ch not in ("fr", "oai"):
            print("'fr' = free | 'oai' = OpenAI")
            emb_ch = input(">>> Embedding: ").lower()
        return emb_ch
    
    def query_strat_ch(self):
        print("'tlk' = talk | 'pg' = paragraph | 'cl' = cluster")
        strat_ch = input(">>> Strategy: ").lower()
        while strat_ch not in ("tlk", "pg", "cl"):
            print("'tlk' = talk | 'pg' = paragraph | 'cl' = cluster")
            strat_ch = input(">>> Strategy: ").lower()
        return strat_ch
    
    def chres(self, choice):
        if not choice or choice not in ["q", "e"]:
            print("'q' = quotes | 'e' = explain")
            choice = input(">>> Response type: ").lower()
            while choice not in ("q", "e"):
                print("'q' = quotes | 'e' = explain")
                choice = input(">>> Response type: ").lower()
        self.prompt_mode = choice
    
    def clear_console(self):
        if os.name == "nt":
            _ = os.system("cls")
        else:
            _ = os.system("clear")
    
    def run(self):
        self.loop()


ConferenceRAG().run()