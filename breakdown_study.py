import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from lmfit.models import ConstantModel


def br_plot(csv_path):
    df = pd.read_csv(csv_path)
    x_name = "Temperature (K)"
    y_name = "Vbd (V)"
    y_unc_name = "Vbd Unc"
    x = [1, 2, 3, 4]
    y = df[y_name].values
    y_unc = df[y_unc_name].values

    horiz_model = ConstantModel()

    avg = horiz_model.fit(y, x=x, weights=1 / (np.asarray(y_unc)))
    br = avg.params["c"].value
    br_err = avg.params["c"].stderr

    residuals = avg.residual

    fit_label = (
        f"Average Uncertainty\n"
        f"Fit: {br:.3f} ± {br_err:.2g}\n"
        f"Reduced Chi Squared: {avg.redchi:2f}\n"
        f"Average Breakdown: {np.mean(y):.3f} ± {np.mean(y_unc):.2g}\n"
        f"Standard Deviation: {np.std(y):.2g}\n"
    )

    plt.title(f"Breakdowns from the July 24th Noise Scan", fontsize=33, pad=20)
    plt.ylabel("Volts")
    plt.xlabel("LED")

    plt.axhline(y=br, label=fit_label)

    plt.errorbar(x=x, y=y, yerr=y_unc, fmt="o", ms=5, lw=1, capsize=5)
    plt.legend(loc="lower center", fontsize=33 - 18)
    plt.grid()
    plt.show()
    # Residuals


path = input("Enter CSV path: ")
br_plot(path)
