from load_data import Data
import numpy as np
import pandas as pd
import argparse
import csv
import json

def seek_entity_description(entity_set):
    formal_knowledge = {"/m/02jxk":"European_Union_Member_States"}
    des_2 = pd.read_csv('./complementary.csv')
    description = pd.read_csv('./save.csv')
    des_set = []
    is_available_set = []
    for i in range(len(entity_set)):
        #entity_des = {"entity_id": None, "description": None}
        #entity_des["entity_id"] = entity_set[i]
        if entity_set[i] in formal_knowledge:
            #entity_des["description"] = formal_knowledge[entity_set[i]]
            entity_des = formal_knowledge[entity_set[i]]
            des_set.append(entity_des)

        else:
            try:
               result = description[description['Entity_id']==entity_set[i]].iloc[0]
               if result["description"] != "":
                   #entity_des["description"] = result["description"]
                   entity_des = str(result["description"]).split("\n")[0]
                   des_set.append(entity_des)
               else:
                   print("no description found in file1")
           
            except:
                try:
                    result = des_2[des_2['Entity_id']==entity_set[i]].iloc[0]
                    if result["description"] != "":
                        entity_des = str(result["description"]).split(".")[0]
                        des_set.append(entity_des)
                    else:
                        print("no description found in file2")
                except:
                    is_available_set.append(entity_set[i])
                    print('no file found')
            

    #des_set = json.dumps(des_set)
    #f = open('entity_description.json', 'w')
    #f.write(des_set)
    des_set = pd.DataFrame(des_set)
    des_set.to_csv('./entity_description.csv', encoding='utf-8', index=False, header=None)
    is_available_set = pd.DataFrame(is_available_set)
    is_available_set.to_csv('./entity_is_availble.csv', encoding='utf-8', index=False, header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--path", type=str, default="../../TuckER", nargs="?",
                    help="Which model path")
    args = parser.parse_args()
    dataset = args.dataset
    path = args.path
    data_dir = "%s/data/%s/" % (path, dataset)
    print(data_dir)
    d = Data(data_dir=data_dir, reverse=False)
    print(len(d.entities))
    seek_entity_description(d.entities)


