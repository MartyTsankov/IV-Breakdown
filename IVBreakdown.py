import numpy as np

import uuid
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model, minimize
from pathlib import Path
from lmfit.models import ConstantModel


# --- 1. a plain Gaussian model ---
def gaussian(x, amp, cen, sigma):
    """Amp = peak height, cen = centre, sigma = std-dev."""
    return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))


gmodel = Model(gaussian)


def expo_then_log(V, V0, b, A, c):
    """
    • V < V0 : baseline b
    • V ≥ V0 : y = b * (1 + (V - V0)/K)**A

      - initial slope on log-y axis  =  A/K
      - long-range growth            ~  A*log(V)
    """

    return np.where(V > V0, b * np.power(V - V0, A) + c, c)


def log(V, V0, b, A):
    return np.where(V > V0, b * np.power(V - V0, A), 0)


def chi_log(params, V, data, weights):
    # Use your existing 'log' function as the model
    model = log(V, **params)

    # Calculate the weighted residuals. lmfit weights are 1/uncertainty.
    residual = (data - model) * weights
    # Calculate chi-squared

    chi_squared = np.sum(residual**2)
    # Determine degrees of freedom

    n_varys = len([p for p in params.values() if p.vary])
    degrees_of_freedom = len(data) - n_varys

    # Calculate reduced chi-squared and its log
    reduced_chi_squared = chi_squared / degrees_of_freedom

    return np.log(reduced_chi_squared)


def sys_unc(value, is_current):
    if is_current:
        i = value
        if i <= 2e-9:
            return [0.01 * i, 2e-12]
        elif i <= 2e-8:
            return [0.004 * i, 2e-12]
        elif i <= 2e-7:
            return [0.003 * i, 2e-10]
        elif i <= 2e-6:
            return [0.002 * i, 2e-10]
        elif i <= 2e-5:
            return [0.001 * i, 2e-8]
        elif i <= 2e-4:
            return [0.001 * i, 2e-8]
        else:
            return [0.001 * i, 2e-6]
    else:  # voltage branch
        v = value
        return [0.15 * v, 5e-3]


def br_plot(volt, err, min, low, high):
    x = list(range(0, len(volt)))
    horiz_model = ConstantModel()
    for i in range(len(err)):
        if err[i] == 0:
            err[i] = 1

    avg = horiz_model.fit(volt, x=x, weights=1 / (np.asarray(err)))
    br = avg.params["c"].value
    br_err = avg.params["c"].stderr
    y_br = np.full_like(x, br, dtype=float)
    residuals = avg.residual
    graph_id = uuid.uuid4().hex

    fit_label = (
        f"Fit: {br:.2f} ± {br_err:.2g}\n"
        f"Average Breakdown: {np.mean(volt):.2f}\n"
        f"Standard Deviation: {np.std(volt):.2g}\n"
        f"Reduced Chi Squared: {avg.redchi:2g}\n"
        f"Fit Range: {min}V to {low}V - {high}V\n"
        f"ID: {graph_id}"
    )
    plt.title(f"Breakdown of {Path(p).stem}", fontsize=33, pad=20)
    plt.ylabel("Volts")
    plt.xlabel("Index")
    plt.plot(x, y_br, label=fit_label)
    plt.errorbar(x, volt, yerr=err, fmt="o", ms=5, lw=1)
    plt.legend(loc="lower left", fontsize=33 - 18)
    plt.grid()
    plt.show()
    # Residuals

    plt.title("Residuals", fontsize=33, pad=20)
    plt.ylabel("Volts")
    plt.xlabel("Index")
    plt.plot(x, residuals, "o", ms=5, color="gray", label=f"ID: {graph_id}")
    plt.axhline(0, color="red", linestyle="--")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


def data(p, fix, r):
    labels = ["Date", "Time", "I_1", "I_2", "Voltage", "IDK"]
    filename = Path(p).stem
    basename = filename.split("-")
    bias = basename[1].strip("V")
    bias = float(bias)

    DF1 = pd.read_csv(p, names=labels, sep="\\s+")

    I = []
    V = []

    for i, row in DF1.iterrows():
        if row.iloc[2] != 0:
            I.append(row.iloc[2])
            V.append(row.dropna().iloc[-1])

    V_max = np.argmax(V)
    V_up = np.asarray(V[fix * 10 : V_max + 1])
    V_up += bias
    V_down = np.asarray(V[V_max:])
    V_down += bias
    I_up = np.asarray(I[fix * 10 : V_max + 1])
    I_down = np.asarray(I[V_max:])

    V_shape = []
    V_check = []
    I_shape = []
    I_check = []
    for i in range(len(V_up) - 1):
        if V_up[i] == V_up[i + 1]:
            V_check.append(V_up[i])
            I_check.append(I_up[i])
        else:
            V_check.append(V_up[i])
            I_check.append(I_up[i])
            V_shape.append(V_check)
            I_shape.append(I_check)
            V_check = []
            I_check = []

    Vd_shape = []
    Vd_check = []
    Id_shape = []
    Id_check = []

    for i in range(len(V_down) - 1):
        if V_down[i] == V_down[i + 1]:
            Vd_check.append(V_down[i])
            Id_check.append(I_down[i])
        else:
            Vd_check.append(V_down[i])
            Id_check.append(I_down[i])
            Vd_shape.append(Vd_check)
            Id_shape.append(Id_check)
            Vd_check = []
            Id_check = []

    ydata = np.array([np.mean(row) for row in I_shape])

    for i in range(0, len(ydata)):
        if ydata[i] > 10**10:
            for j in range(len(I_shape[i])):
                if I_shape[i][j] > 10**10:
                    I_shape[i][j] = I_shape[i][j - 1]

    num_points_up = len(V_up)
    num_points_down = len(V_down)

    if r:
        return Vd_shape[::-1], Id_shape[::-1], num_points_down, basename
    return V_shape, I_shape, num_points_up, basename


def plot_hist(hist_list, max_current, alpha=0.5):
    bins = 27
    for p, fix in hist_list:
        br_index = plot_breakdown(p, fix, False)
        V, I, num_points, basename = data(p, fix)
        if br_index is not None:
            I = I[: br_index - 1]
            currents = []
            for i in range(len(I)):
                for j in range(len(I[i])):
                    currents.append((I[i][j]))

            currents = I[:br_index].ravel()
        else:
            currents = I
        currents = currents[currents <= max_current]

        counts, bins = np.histogram(currents, bins=27)  # Generate histogram data

        bin_midpoints = (bins[:-1] + bins[1:]) / 2  # Calculate bin midpoints
        mean = np.sum(bin_midpoints * counts) / np.sum(counts)  # Calculate the mean
        # --- turn the histogram into x–y points for fitting ---
        centers = 0.5 * (bins[:-1] + bins[1:])  # x-values
        mask = counts > 0  # skip empty bins

        # --- define & fit a Gaussian with lmfit ---
        from lmfit import Model

        def gaussian(x, amp, cen, sigma):
            return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))

        gmod = Model(gaussian)
        params = gmod.make_params(
            amp=counts.max(), cen=centers[counts.argmax()], sigma=np.std(currents)
        )

        result = gmod.fit(
            counts[mask], params, x=centers[mask], weights=1.0 / np.sqrt(counts[mask])
        )  # 1/√N Poisson weights

        # Optional: plot the histogram

    plt.show()


def plot_hist_check(hist_list, max_current, alpha=0.5):
    fs = 33
    for p, fix in hist_list:
        br_index = plot_breakdown(p, fix, False)
        V, I, num_points, basename = data(p, fix)

        currents = I[:br_index].ravel()
        currents = currents[currents <= max_current]
        currents *= 1e6
        counts, bins = np.histogram(currents, bins=27)

        # ----- Gaussian fit -----
        centers = 0.5 * (bins[:-1] + bins[1:])
        mask = counts > 0  # ignore empty bins

        gmod = Model(gaussian)
        params = gmod.make_params(
            amp=counts.max(), cen=centers[counts.argmax()], sigma=np.std(currents)
        )
        result = gmod.fit(
            counts[mask], params, x=centers[mask], weights=1.0 / np.sqrt(counts[mask])
        )

        # ----- plot -----
        bin_width = bins[1] - bins[0]
        plt.bar(
            centers,
            counts,
            width=bin_width,
            alpha=alpha,
            label=f"{basename[0]} ({basename[4]}) data",
        )

        x_fit = np.linspace(centers.min(), centers.max(), 400)
        plt.plot(
            x_fit,
            result.eval(x=x_fit),
            "-",
            lw=2,
            label=(
                f"{basename[0]} ({basename[4]}) fit \n Gaussian: "
                f"μ={result.params['cen'].value:.3g}±{result.params['cen'].stderr:.2g}\n"
                f"σ={result.params['sigma'].value:.3g}±{result.params['sigma'].stderr:.2g}"
            ),
        )

        plt.legend(loc="upper left", fontsize=fs - 17)

    plt.xlabel("Current (μ Amps)", fontsize=fs)
    plt.ylabel("Counts", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid()
    plt.show()


def plot_breakdown(
    p, fix, r, line, figure, show=True, initial_window_size=25, threshold=2
):
    V, I, num_points, basename = data(p, fix, r)
    unc_list = []
    N = []
    for row in I:
        unc_row = [np.asarray(sys_unc(i, is_current=True)) for i in row]
        unc_list.append(unc_row)
        N.append(len(row))

    vunc_list = []
    for row in V:
        vunc_row = [np.asarray(sys_unc(v, is_current=False)) for v in row]
        vunc_list.append(vunc_row)
    N = np.asarray(N)

    gain_var_list = []
    offset_var_list = []

    for i, row in enumerate(I):
        unc_row = [sys_unc(val, is_current=True) for val in row]
        sum_gain_u = sum(u[0] for u in unc_row)
        sum_offset_u = sum(u[1] for u in unc_row)
        gain_var_list.append((sum_gain_u / N[i]) ** 2)
        offset_var_list.append((sum_offset_u / N[i]) ** 2)
    # rand_u = np.std(I, axis=1, ddof=1) / np.sqrt(N)

    rand_var = np.array([np.var(np.asarray(row), ddof=1) / len(row) for row in I])
    gain_var = np.array(gain_var_list)
    offset_var = np.array(offset_var_list)
    yerr = np.sqrt(rand_var + gain_var + offset_var)

    # step = np.diff(xdata).mean().round(3)
    xdata = np.array([row[0] for row in V])
    ydata = np.array([np.asarray(row).mean() for row in I])
    y_abs_data = np.abs(ydata)
    if r:
        ydata = np.abs(ydata)
    # --- HYBRID APPROACH: Use your original chi-squared method to find the initial guess ---

    # 1. First, establish the baseline noise level from the initial flat part of the data.
    horiz_model = ConstantModel()
    baseline_fit = horiz_model.fit(
        y_abs_data[:initial_window_size],
        x=xdata[:initial_window_size],
        weights=1.0 / yerr[:initial_window_size],
    )

    baseline_guess = baseline_fit.params["c"].value
    # 2. Now, find the point where the data significantly deviates from this baseline.
    chisq_reduced = np.full_like(y_abs_data, np.nan)
    for i in range(initial_window_size, len(y_abs_data)):
        # Calculate chi-squared for all points up to 'i' against the constant baseline model
        resid = y_abs_data[: i + 1] - baseline_guess
        chisq = np.sum((resid / yerr[: i + 1]) ** 2)
        dof = i + 1  # Degrees of freedom
        chisq_reduced[i] = chisq / dof if dof > 0 else 0

    # 3. The "br" is the point just before the chi-squared value crosses the threshold.
    # We find the first index where the condition is met.
    # The threshold is arbitrary, but since it's just a guess for the real fit it's ok

    if line:
        baseline_fit2 = horiz_model.fit(
            y_abs_data,
            x=xdata,
            weights=1.0 / yerr,
        )

        baseline_val = baseline_fit2.params["c"].value
        baseline_err = baseline_fit2.params["c"].stderr

        fig, ax1 = plt.subplots()
        fs = 33

        ramp = ""
        if r:
            ramp = "Ramp Down"
        else:
            ramp = "Ramp Up"
        fit_legend_label = (
            "Fit Model: " + r"$y(V) = b$" + "\n"
            f"c = {baseline_val:.2g} ± {baseline_err:.1g}\n"
            f"{ramp}"
        )

        ax1.errorbar(
            xdata,
            ydata,
            yerr=yerr,
            fmt="o",
            ms=5,
            lw=1,
            label=Path(p).stem,
            color="#1f77b4",
            zorder=3,
        )
        ax1.set_yscale("log")
        ax1.set_xlabel("Bias Voltage [Volts]", fontsize=fs)
        ax1.set_ylabel("Current [Amps]", fontsize=fs, color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=fs)
        ax1.tick_params(axis="x", labelsize=fs)
        ax1.set_ylim(1e-11, 1e-3)
        ax1.grid(True)
        v_plot = np.linspace(xdata.min(), xdata.max(), 400)
        ax1.plot(
            v_plot,
            baseline_fit2.eval(x=v_plot),
            "-",
            lw=2.5,
            color="orange",
            label=fit_legend_label,
            zorder=4,
            alpha=1.0,
        )
        ax1.legend(loc="upper left", fontsize=fs - 18)
        plt.title(f"Baseline of {Path(p).stem}", fontsize=33, pad=20)
        plt.show()

        return xdata[-1]

    br_candidates = np.where(chisq_reduced >= threshold)[0]
    if br_candidates.size > 0:
        br_row = br_candidates[0]

    baseline_fit2 = horiz_model.fit(
        y_abs_data[:br_row],
        x=xdata[:br_row],
        weights=1.0 / yerr[:br_row],
    )
    baseline_guess = baseline_fit2.params["c"].value
    baseline_error = baseline_fit2.params["c"].stderr

    pw_model = Model(expo_then_log)
    weights = 1.0 / yerr
    V0_guess = xdata[br_row]
    A_guess = 1.7
    # Create the parameter set with our improved guesses.
    params1 = pw_model.make_params(
        b=baseline_guess, V0=V0_guess, A=A_guess, c=baseline_guess
    )

    # --- CHANGED: More flexible parameter bounds ---
    # Allow the baseline to be negative or positive, as noise can cause this.
    params1["b"].set(value=baseline_guess * 10, vary=True, min=1e-13)
    params1["c"].set(value=baseline_guess, vary=True, min=1e-13)
    # dI must be positive.
    params1["V0"].set(min=xdata.min(), max=xdata.max())
    params1["A"].set(min=0, max=50)  # Allow k to be much smaller or larger

    # Range
    min = xdata.min()
    max = xdata.max()
    low = 29
    high = 31
    V0_guess_low = np.random.uniform(min, min, 50)
    V0_guess_high = np.random.uniform(low, high, 50)

    br_list = []
    err_list = []

    for j in range(len(V0_guess_high)):
        xrange = []
        yrange = []
        wrange = []
        for i in range(len(xdata)):
            if xdata[i] > V0_guess_low[j] and xdata[i] < V0_guess_high[j]:
                xrange.append(xdata[i])
                yrange.append(y_abs_data[i])
                wrange.append(weights[i])

        # Then the fit is performed on the *entire* dataset:

        fit1 = pw_model.fit(yrange, params1, V=xrange, weights=wrange, method="leastsq")

        # print(fit1.fit_report())
        V0_val = fit1.params["V0"].value

        xrange = []
        yrange = []
        wrange = []
        yerr_range = []
        for i in range(len(xdata)):
            if xdata[i] > V0_guess_low[j] and xdata[i] < V0_guess_high[j]:
                xrange.append(xdata[i])
                yrange.append(y_abs_data[i])
                wrange.append(weights[i])
                yerr_range.append(yerr[i])
        params2 = fit1.params
        fit = pw_model.fit(yrange, params2, V=xrange, weights=wrange, method="leastsq")

        """
        params3 = fit2.params

        V0_val = fit2.params["V0"].value
        V0_err = fit2.params["V0"].stderr

        xrange = []
        yrange = []
        wrange = []
        yerr_range = []
        if xdata[i] > V0_guess_low[j] and xdata[i] < V0_guess_high[j]:
            xrange.append(xdata[i])
            yrange.append(y_abs_data[i])
            wrange.append(weights[i])
            yerr_range.append(yerr[i])
        fit = pw_model.fit(yrange, params3, V=xrange, weights=wrange, method="leastsq")
        """

        # --- CALCULATE STATISTICALLY MEANINGFUL CHI-SQUARED ---
        residuals_unweighted = np.asarray(yrange) - fit.best_fit
        chisq_stat = np.sum((residuals_unweighted / yerr_range) ** 2)
        dof = len(yrange) - fit.nvarys
        red_chisq_stat = chisq_stat / dof if dof > 0 else 0.0
        params = fit.params

        V0_val = params["V0"].value
        V0_err = fit.params["V0"].stderr if params["V0"].stderr is not None else 0.0
        b_val = params["b"].value
        b_err = params["b"].stderr if params["b"].stderr is not None else 0.0
        A_val = params["A"].value
        A_err = params["A"].stderr if params["A"].stderr is not None else 0.0
        c_val = params["c"].value
        c_err = params["c"].stderr if params["c"].stderr is not None else 0.0
        red_chisq = red_chisq_stat
        # print(fit.fit_report())

        # figure
        if figure is False:
            fig, ax1 = plt.subplots()
            fs = 33

            ramp = ""
            if r:
                ramp = "Ramp Down"
            else:
                ramp = "Ramp Up"
            fit_legend_label = (
                "Fit Model: " + r"$y(V)=c\,(V - V_{0})^{A} + b$" + "\n"
                f"c = {b_val:.2g} ± {b_err:.1g}\n"
                f"b (Fixed) = {c_val:.2g} ± {baseline_error:.1g}\n"
                f"A = {A_val:.2g} ± {A_err:.1g}\n"
                f"$\\chi^2_\\nu$ = {red_chisq:.2g}\n"
                f"Range: {V0_guess_low[j]:.2f} - {V0_guess_high[j]:.2f}\n"
                f"{ramp}"
            )
            # IV points with proper σ
            ax1.errorbar(
                xdata,
                ydata,
                yerr=yerr,
                fmt="o",
                ms=5,
                lw=1,
                label=Path(p).stem,
                color="#1f77b4",
                zorder=3,
            )
            ax1.set_yscale("log")
            ax1.set_xlabel("Bias Voltage [Volts]", fontsize=fs)
            ax1.set_ylabel("Current [Amps]", fontsize=fs, color="#1f77b4")
            ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=fs)
            ax1.tick_params(axis="x", labelsize=fs)
            ax1.set_ylim(1e-11, 1e-3)
            ax1.grid(True)

            # Generate a dense set of x-values for a smooth plot of the fit
            v_dense = np.linspace(xdata.min(), xdata.max(), 400)
            # The line below was causing the artificial vertical line. We plot the real fit now.
            v_plot = []
            for x in v_dense:
                if x < V0_guess_high[j]:
                    v_plot.append(x)
            ax1.plot(
                v_plot,
                fit.eval(V=v_plot),
                "-",
                lw=2.5,
                color="orange",
                label=fit_legend_label,
                zorder=4,
                alpha=1.0,
            )
            ax1.axvline(
                V0_val,
                ls="--",
                lw=2,
                color="darkred",
                label=f"Breakdown: ({V0_val:.2f} ± {V0_err:.2g} V)",
            )
            ax1.axvspan(
                V0_guess_low[j],
                V0_guess_high[j],
                color="grey",
                alpha=0.2,
                label=f"Fit Range: {V0_guess_low[j]:.2f} - {V0_guess_high[j]:.2f}",
            )

            # Combined legend with improved positioning
            ax1.legend(loc="lower right", fontsize=fs - 18)
            plt.title(f"Breakdown of {Path(p).stem}", fontsize=33, pad=20)
            plt.show()
        # figure

        br_list.append(V0_val)
        err_list.append(V0_err)
    br_plot(br_list, err_list, min, low, high)
    return V0_val


hist = input("Hist? (y/N) ").lower() == "y"

if hist:
    hist_list = []
    for i in range(int(input("How many IVs? "))):
        p = str(input("Enter file path: "))
        fix = int(input("Enter fix: "))
        hist_list.append((p, fix))

    plot_hist_check(hist_list, 5e-7)
else:
    p = str(input("Enter file path: "))
    fix = int(input("Enter fix: "))
    r = input("Ramp down? (y/N) ").lower() == "y"
    line = input("Baseline? (y/N) ").lower() == "y"
    figure = input("Average Breakdown? (y/N) ").lower() == "y"
    br = plot_breakdown(p, fix, r, line, figure)
