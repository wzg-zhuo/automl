import datetime
import pickle
import logging
import pandas as pd
import yaml
import time
import json

def get_logger():
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger=logging.getLogger(__name__)
    return logger

def write_append(path,content_list):
    with open(path,'a') as f:
        for item in content_list:
            f.write(item)
        f.write('\n')

def write_log(report,f1):
    _list=[]
    _list.append("*"*30+"Classification report"+"*30"+"\n")
    _list.append("model_time:"+str(datetime.datetime.now().ctime()),)
    _list.append(report+"\n")
    _list.append("f1_score:"+str(f1)+"\n")
    log_path="log/result_log.txt"
    write_append(log_path,_list)
    logger=__logger()
    logger.info("log saved to {}".format(log_path))



def logad_pkl(data_path):
    with open(data_path,'rb') as f:
        data=pickle.loads(f)
        return data


def get_config(yaml_file):
    with open(yaml_file,"r",encoding="utf-8") as f:
        data=f.read()
    config=yaml.load(data)
    return config



