import glob
import json
import pandas as pd
from flask import Flask, render_template
import math
import numpy as np

app = Flask(__name__)

def read_params(model_name):
    #model_name=args.model_name
    data=[]
    for hist_fname in sorted(glob.glob(model_name+"__*.history.json")):
        with open(hist_fname) as f:
            epoch_list,score_dicts=json.load(f)
        params=hist_fname.replace(".history.json","").split("__")
        assert params[0]==model_name
        item_data_dict={}
        for pname_val in params[1:]:
            pname,val=pname_val.rsplit("_",1)
            val=float(val)
            item_data_dict["param_"+pname]=val
        for score,vals in score_dicts.items():
            item_data_dict["sc_"+score]=vals[-1]
        data.append(item_data_dict)
    all_columns=list(sorted(set(k for item_data_dict in data for k in item_data_dict)))
    elems=[]
    for item_data_dict in data:
        elems.append(list(item_data_dict.get(k,None) for k in all_columns))
    df=pd.DataFrame.from_records(elems,columns=all_columns)

    score_cols=[col for col in all_columns if col.startswith("sc_")]
    for scol in score_cols:
        df["log_"+scol]=np.log(df[scol]) #make log ranges
    return df

@app.route('/')
def root():
    global df,args
    df=read_params(args.prefix)
    df=df.filter(regex="param_.*|sc_({0})|log_sc_({0})".format("|".join(args.scores.split(","))))
    return render_template('index.html',datatable=df.to_html(),model_prefix=args.prefix)

@app.route('/data.tsv')
def data_csv():
    global df,args
    return df.to_csv(sep="\t",index=False)

#df=read_params("taito_models/model.fin.WEmbDepPredictor2/model.fin.WEmbDepPredictor2")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--prefix",help="Model file prefix to gather the .history.json files")
    parser.add_argument("--port",default=5957,type=int,help="Port %(default)d")
    parser.add_argument("--scores",default="val_loss",help="Scores to include, comma-separated like 'loss,val_loss'. Their log versions will be included as well.  Default: %(default)s")
    args = parser.parse_args()

    df=None
    
    app.run(debug=False,port=args.port)
