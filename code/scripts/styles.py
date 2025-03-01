import matplotlib.pyplot as plt

def set_plot_style():
    plt.style.use("seaborn-muted")  # Use a predefined style
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 300,   # for higher resolution 
        "axes.grid": True
    })
