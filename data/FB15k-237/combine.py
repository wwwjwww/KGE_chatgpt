import os
import pandas as pd

path = "./relations.txt"
path1 = "./save.txt"

relation = pd.read_csv(path, header=None)
content = pd.read_csv(path1, sep='\t', header=None)
combine = pd.concat([relation, content], axis=1)
combine.to_csv('chatgpt_relations_descriptions.txt', sep='\t', header=None, index=False)
