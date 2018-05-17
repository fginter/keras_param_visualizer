# Intro
A little script to help in parameter selection. Parameters and their corresponding score are plotted in a parallel axes plot which is interactive and allows you to (attempt to) spot the good parameter ranges.

![image](https://github.com/fginter/keras_param_visualizer/blob/master/param_vis_ss.png)

# How to use

# Keras

Dump as json the training history you get from `model.fit()` and encode the parameters in the name like so:

    parameters={"lrate":0.01,"dropout":0.15,"embedding_width":200} #have a dictionary with your parameters
    param_string="__".join("{}_{}".format(k,v) for k,v in kwargs.items())
    #gives you "lrate_0.01__dropout_0.15__embedding_width_200" ... always two underscores between parameters
    (...)
    hist=model.fit(...) #...fit your model and get history
    with open("logdir/my_model__"+param_string+".history.json","w") as f:
        json.dump((hist.epoch,hist.history),f)
    
# Checking your parameters

Once you have these `.history.json` files, you can run this code as follows:

    python3 param_vis.py --prefix "logdir/my_model"
    
This will look for all files called `logdir/my_model__*.history.json` and serve a page on port `5957`
(you can select your own port with `--port 6666`) with the interactive plot. If you run it on a different server just do

    ssh -L 5957:localhost:5957 serveraddress
    
and point your browser to `http://localhost:5957`

# Scores

All scores found in the hist file are available, together with their log versions. Default is to use `val_loss`, but you can specify your own:

    python3 param_vis.py --prefix "logdir/my_model" --scores val_loss,loss
    
