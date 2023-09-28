from load_data import Data
import numpy as np
import pandas as pd
import argparse
import csv

def seek_entity_description(entity_set):
    description = pd.read_csv('./FB15K-237_Entities.csv', sep=',', quotechar='"')
    des_set = []
    is_available_set = []
    for i in range(len(entity_set)):
        try:
           result = description[description['Entity_id']==entity_set[i]].iloc[0]
           if result["description"] != "":
               des_set.append(result["description"])
           else:
               if result["Entity_name"] != "":
                   des_set.append(result["Entity_name"])
               else:
                   print("no description found")

        except:
            is_available_set.append(entity_set[i])

    des_set = pd.DataFrame(des_set)
    des_set.to_csv('./FB15k-237-entity_simple_description.csv', encoding='utf-8', index=False, header=None)
    is_available_set = pd.DataFrame(is_available_set)
    is_available_set.to_csv('./FB15k-237-entity_is_availble.csv', encoding='utf-8', index=False, header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--path", type=str, default="../TuckER", nargs="?",
                    help="Which model path")
    args = parser.parse_args()
    dataset = args.dataset
    path = args.path
    data_dir = "%s/data/%s/" % (path, dataset)
    d = Data(data_dir=data_dir, reverse=False)
    seek_entity_description(d.entities)


