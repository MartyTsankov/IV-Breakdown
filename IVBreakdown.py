import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model
from pathlib import Path
from lmfit.models import ConstantModel


# --- 1. a plain Gaussian model ---
def gaussian(x, amp, cen, sigma):
    """Amp = peak height, cen = centre, sigma = std-dev."""
    return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))


gmodel = Model(gaussian)


def expo_then_log(V, V0, b, A, K):
    """
    • V < V0 : baseline b
    • V ≥ V0 : y = b * (1 + (V - V0)/K)**A

      - initial slope on log-y axis  =  A/K
      - long-range growth            ~  A*log(V)
    """
    out = np.full_like(V, b, dtype=float)
    mask = V >= V0
    out[mask] = b * np.power((V[mask] - V0) / K + 1.0, A)
    return out


def log(V, V0, b, A, K):
    return np.log(expo_then_log(V, V0, b, A, K) + 1e-20)


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


def data(p, fix):
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
    V_down = np.asarray(V[:V_max:])
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

    ydata = np.array([np.mean(row) for row in I_shape])

    for i in range(0, len(ydata)):
        if ydata[i] > 10**10:
            for j in range(len(I[i])):
                if I[i][j] > 10**10:
                    I[i][j] = I[i][j - 1]

    num_points = len(V_up)

    return V_shape, I_shape, num_points, basename


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


def plot_breakdown(p, fix, show=True, initial_window_size=25, threshold=1):
    V, I, num_points, basename = data(p, fix)
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

    logy = np.log(y_abs_data)
    logyerr = yerr / ydata

    pw_model = Model(log)
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
    br_candidates = np.where(chisq_reduced >= threshold)[0]
    if br_candidates.size > 0:
        br_row = br_candidates[0]
    else:
        # Fallback: if threshold is never crossed, guess the midpoint.
        br_row = len(xdata) // 2
        print("Warning: Chi-squared threshold not crossed. Guessing midpoint for V0.")

    baseline_fit2 = horiz_model.fit(
        y_abs_data[:br_row],
        x=xdata[:br_row],
        weights=1.0 / yerr[:br_row],
    )
    baseline_guess = baseline_fit2.params["c"].value
    weights = 1.0 / (logyerr**2)
    V0_guess = xdata[br_row]
    print(V0_guess)
    K_guess = 1.0
    A_guess = 1.0
    # Create the parameter set with our improved guesses.
    params1 = pw_model.make_params(b=baseline_guess, V0=V0_guess, A=A_guess, K=K_guess)

    # --- CHANGED: More flexible parameter bounds ---
    # Allow the baseline to be negative or positive, as noise can cause this.
    params1["b"].set(value=baseline_guess, vary=False, min=1e-13)
    # dI must be positive.
    params1["V0"].set(min=xdata.min(), max=xdata.max())
    params1["A"].set(min=0, max=50)  # Allow k to be much smaller or larger
    params1["K"].set(min=1e-3, max=20)

    # Then the fit is performed on the *entire* dataset:
    fit1 = pw_model.fit(
        logy, params1, V=xdata, weights=weights, method="leastsq", max_nfev=100000
    )

    params2 = fit1.params
    params2["b"].set(vary=False)
    fit = pw_model.fit(logy, params2, V=xdata, weights=weights, method="leastsq")

    print(fit.fit_report(min_correl=0.5))

    params = fit.params
    V0_val = params["V0"].value
    V0_err = params["V0"].stderr if params["V0"].stderr is not None else 0.0
    b_val = params["b"].value
    b_err = params["b"].stderr if params["b"].stderr is not None else 0.0
    A_val = params["A"].value
    A_err = params["A"].stderr if params["A"].stderr is not None else 0.0
    K_val = params["K"].value
    K_err = params["K"].stderr if params["K"].stderr is not None else 0.0
    red_chisq = fit.redchi

    # Create figure with adjusted layout
    fig, ax1 = plt.subplots()
    fs = 33

    fit_legend_label = (
        f"Fit Model:\n"
        f"b = {b_val:.2g} ± {b_err:.1g}\n"
        f"A = {A_val:.2g} ± {A_err:.1g}\n"
        f"K = {K_val:.2g} ± {K_err:.1g}\n"
        f"$\\chi^2_\\nu$ = {red_chisq:.2g}"
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
    v_dense = np.linspace(xdata.min(), xdata.max(), 500)
    # The line below was causing the artificial vertical line. We plot the real fit now.
    v_plot = []
    for i in v_dense:
        if i < V0_val + 3.5:
            v_plot.append(i)
    v_plot = np.asarray(v_plot)
    log_fit_curve = fit.eval(V=v_plot)
    fit_curve = np.exp(log_fit_curve)
    ax1.plot(
        v_plot,
        fit_curve,
        "-",
        lw=2.5,
        color="gray",
        label=fit_legend_label,
        zorder=4,
    )
    ax1.axvline(
        V0_val,
        ls="--",
        lw=2,
        color="darkred",
        label=f"Breakdown: ({V0_val:.2f} ± {V0_err:.2f} V)",
    )
    # Combined legend with improved positioning
    ax1.legend(loc="upper left", fontsize=fs - 15)
    plt.title(f"Breakdown of {Path(p).stem}", fontsize=33, pad=20)
    plt.show()


hist = input("Hist? (y/N)").lower() == "y"

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
    plot_breakdown(p, fix)
