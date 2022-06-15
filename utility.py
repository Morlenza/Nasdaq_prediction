import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np

def gain(D, D_pred):
    growth = D_pred > 0
    decline = D_pred < 0
    return D[growth].sum() - D[decline].sum()
gain.__doc__ = "Restituisce il guadagno (in euro)"

def roi(D, D_pred, open):
    mean_open = open.reindex_like(D).mean()
    return gain(D, D_pred)/ mean_open
roi.__doc__ = "Restituisce il guadagno in percentuale"

def print_eval(X, y, model, open):
    preds = model.predict(X)
    print("Gain: {:.2f}$".format(gain(y, preds)))
    print(" ROI: {:.3%}".format(roi(y, preds, open)))
print_eval.__doc__ = "Calcola i valori predetti dato un modello e stampa a video Gain e ROI"
    
def prepare_data(features, delta):
    X = pd.DataFrame(features)
    X.dropna(inplace=True)
    y = delta.reindex_like(X)
    return X, y
prepare_data.__doc__ = "Restituisce X e y data una lista di features e un parametro delta"

def plot_model_on_data(X, y, model=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(X, y)
    if model is not None:
        xlim, ylim = plt.xlim(), plt.ylim()
        line_x = np.linspace(xlim[0], xlim[1], 100)
        line_y = model.predict(line_x[:, None])
        plt.plot(line_x, line_y, c="red", lw=3)
        plt.xlim(xlim); plt.ylim(ylim)
    plt.grid()
    plt.xlabel("Temperatura (°C)"); plt.ylabel("Consumi (GW)")
plot_model_on_data.__doc__ = "Visualizza un grafico che mostra l'applicazione della LinearRegression sui dati"