import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        "font.size": 10,         # Base font size
        "axes.titlesize": 12,    # Title size
        "axes.labelsize": 11,    # Label size
        "xtick.labelsize": 9,    # Tick labels
        "ytick.labelsize": 9,
        "legend.fontsize": 10,   # Legend font size
        "figure.figsize": (5.3, 4),  # Default figure size (adjustable height)
        "savefig.dpi": 600,      # High-quality figures
        "axes.grid": False,       # Enable grid
        "axes.linewidth": 0.5,     # Axis border thickness
        "xtick.major.size": 2,   # Major tick size
        "ytick.major.size": 2,
        "xtick.major.width": 0.4, # Tick width
        "ytick.major.width": 0.4,
        "lines.linewidth": 1.5,   # Line thickness
        "lines.markersize": 5     # Marker size
    })

