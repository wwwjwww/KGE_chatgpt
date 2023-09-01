from typing import List
import json
import sys
import time

import openai


class GraphGPT:

    def __init__(self, api_key: str, messages: List = None):
        openai.api_key = api_key
        if messages:
            self.messages = messages
        else:
            self.messages = [
                {"role": "assistant", "content": "Knowledge graph is a set of triplets. Each triplet consists a head entity, a tail entity and the relationship between this two entities. For example, 'Eiffel_tower locatedIn Paris' is a typical triplet where 'Eiffel_tower' is the head entity, 'Paris' is the tail entity, and 'located in' is the relationship between these two entities."}
            ]

    def ask_chat_gpt(self) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        response_content = response['choices'][0]['message']['content']
        return response_content

    def ask_each_triple(self, r1: str):
        self.messages.append({"role": "user", "content": f"'{r1}' is the relation between two entities in a knowledge graph, what does it mean? please reply in one sentence begins with 'the relation indicates'.Do not reply with extra words. "})
        response_content = self.ask_chat_gpt()
        print(response_content)
        self.messages.pop()

        if 'The relation indicates' in response_content:
            return response_content
        else:
            return None

if __name__ == '__main__':

    api_key = "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"

    sim_chatgpt = GraphGPT(api_key=api_key)
    with open('./data/FB15k-237/relations.txt') as f:
        for line in f:
            r = str(line)
            content = sim_chatgpt.ask_each_triple(r)
            with open('./data/FB15k-237/chatgpt_relations_description.txt', 'a') as f1:
                f1.write(r)
                f1.write('\t')
                f1.write(content)
                f1.write('\n')
            time.sleep(20)
