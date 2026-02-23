"""
loss_curves.py — Visualizations for training loss history.
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_loss_landscape(
    loss_history: list,
    switch_tau: list = None,
    out_path: str = "results/loss_landscape.png",
    title: str = "Optimization Trajectory & Loss Landscape",
):
    """
    Advanced 3-panel optimization dashboard (Ultra-Q1 Quality).
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass
    import matplotlib.gridspec as gridspec
    
    if switch_tau is None: switch_tau = []

    T = len(loss_history)
    max_epochs = max(len(h) for h in loss_history)
    mat = np.full((max_epochs, T), np.nan)  # Epochs on Y, Tau on X
    
    # Flatten loss for a continuous curve plot
    continuous_loss = []
    tau_boundaries = [0]
    
    for t, hist in enumerate(loss_history):
        mat[:len(hist), t] = np.maximum(np.abs(hist), 1e-12) # prevent log(0)
        continuous_loss.extend(np.maximum(np.abs(hist), 1e-12))
        tau_boundaries.append(tau_boundaries[-1] + len(hist))

    continuous_loss = np.array(continuous_loss)
    epochs = np.arange(1, len(continuous_loss) + 1)
    
    # Calculate gradient norm proxy (absolute difference in loss)
    grad_proxy = np.abs(np.diff(continuous_loss))
    grad_proxy = np.append(grad_proxy, grad_proxy[-1]) # pad length
    
    # Calculate moving variance of gradient proxy
    window = min(20, max(5, len(continuous_loss)//10))
    if len(continuous_loss) > window:
        smoothed_loss = np.convolve(continuous_loss, np.ones(window)/window, mode='same')
        smoothed_grad = np.convolve(grad_proxy, np.ones(window)/window, mode='same')
    else:
        smoothed_loss = continuous_loss
        smoothed_grad = grad_proxy

    fig = plt.figure(figsize=(20, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.3)
    
    c_loss, c_smooth = '#D1C4E9', '#512DA8'  # Refined purple tones
    c_grad = '#FF8F00'
    
    # --- Panel (a): Global Loss Convergence ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, continuous_loss, color=c_loss, lw=1.5, alpha=0.9, label="Epoch Loss $\\mathcal{L}(\\theta)$")
    ax1.plot(epochs, smoothed_loss, color=c_smooth, lw=2.5, label=f"Smoothed Trajectory (w={window})")
    
    # Mark tau steps
    for b in tau_boundaries[1:-1]:
        ax1.axvline(b, color="gray", linestyle="--", alpha=0.3, lw=1.0)
        
    ax1.set_yscale("log")
    ax1.set_xlabel("Cumulative Training Epochs", fontweight="bold")
    ax1.set_ylabel("MMD Objective Loss (Log Scale)", fontweight="bold")
    ax1.set_title("(a) Global Convergence Trajectory", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls="--")

    # --- Panel (b): Optimization Stability (Gradient Proxy) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, grad_proxy, color='#FFECB3', alpha=0.9, label='Step-wise $\\Delta\\mathcal{L}$')
    ax2.plot(epochs, smoothed_grad, color=c_grad, lw=2.5, label='Smoothed Gradient Proxy $|\\nabla\\mathcal{L}|$')
    
    # Add Exponential decay fit on smoothed loss to show theoretical vs empirical
    # Simple log linear fit on smoothed loss
    try:
        # Avoid zeros for log matching
        valid_idx = smoothed_loss > 1e-10
        if sum(valid_idx) > 5:
            fit_x = epochs[valid_idx]
            y_log = np.log(smoothed_loss[valid_idx])
            p = np.polyfit(fit_x, y_log, 1)
            fit_y = np.exp(np.polyval(p, epochs))
            ax1.plot(epochs, fit_y, color='#D32F2F', linestyle=':', lw=2.5, label=f'Exp Bound: $\\mathcal{{O}}(e^{{{p[0]:.3f} e}})$')
            ax1.legend(loc="upper right", framealpha=0.9) # update legend 
    except Exception as e:
        logger.warning(f"Failed to fit exp curve: {e}")

    ax2.set_yscale("log")
    ax2.set_ylim(max(1e-8, min(smoothed_grad)*0.1), max(grad_proxy)*5)
    ax2.set_xlabel("Cumulative Training Epochs", fontweight="bold")
    ax2.set_ylabel("Optimization Velocity $|\\Delta\\mathcal{L}|$", fontweight="bold")
    ax2.set_title("(b) Loss Gradient Stability", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls="--")

    # --- Panel (c): Epoch-wise Loss Heatmap ---
    ax_heat = fig.add_subplot(gs[2])
    mat_log = np.log10(mat)
    im = ax_heat.imshow(mat_log, aspect="auto", cmap="magma", interpolation="nearest")
    
    ax_heat.set_xlabel("Integration Step $\\tau$", fontweight="bold")
    ax_heat.set_ylabel("Epoch Index", fontweight="bold")
    ax_heat.set_title("(c) Loss Density Landscape", fontsize=14, fontweight="bold")
    
    xticks = list(range(0, T, max(1, T // 5)))
    if T - 1 not in xticks:
        xticks.append(T - 1)
    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels([x + 1 for x in xticks])
    
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.02)
    cbar.set_label("Log₁₀(Loss)", rotation=270, labelpad=15, fontweight="bold")

    fig.suptitle(f"Quantum Kinematics: {title}", fontsize=17, fontweight="bold", y=1.05)
    
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Advanced Q1 loss landscape to {out_path}")
