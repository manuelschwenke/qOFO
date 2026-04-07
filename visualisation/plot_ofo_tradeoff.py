import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

tahoma_path = None
for f in fm.findSystemFonts():
    if 'tahoma' in f.lower():
        tahoma_path = f
        break
prop   = fm.FontProperties(fname=tahoma_path, size=13) if tahoma_path else fm.FontProperties(size=13)
prop14 = fm.FontProperties(fname=tahoma_path, size=13) if tahoma_path else fm.FontProperties(size=13)

# g(w) = w^T G_w w + ∇Φ^T H̃ w  →  scalar: G*w² + c*w,  optimum: w* = -c/(2G)
c      = -1.5   # combined ∇Φ^T H̃
G_vals = [0.5, 1.5, 4.0]
colors = ["#4C8BB5", "#B1BD00", "#E05C5C"]
labels = [r"$G_\mathrm{w}$ small", r"$G_\mathrm{w}$ medium", r"$G_\mathrm{w}$ large"]

w  = np.linspace(-2.5, 5.0, 600)
w2 = np.linspace(0, 5.0, 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# ── Panel 1: Cost function ────────────────────────────────────────────────────
ax = axes[0]
for G, col, lab in zip(G_vals, colors, labels):
    J      = G * w**2 + c * w
    w_star = -c / (2 * G)
    J_star = G * w_star**2 + c * w_star
    ax.plot(w, J, color=col, linewidth=2.2, label=lab)
    ax.plot(w_star, J_star, 'o', color=col, markersize=7, zorder=5)
    ax.axvline(w_star, color=col, linewidth=1.0, linestyle=':', alpha=0.6)

ax.axhline(0, color='gray', linewidth=0.7)
ax.set_xlabel(r'$w^k$', fontproperties=prop)
ax.set_ylabel(
    r'$g(w^k) = (w^k)^\top \mathbf{G}_\mathrm{w} w^k + \nabla f^\top \mathbf{H} w^k$',
    fontproperties=prop)
ax.set_title("Cost Function & Optimum", fontproperties=prop14)
ax.set_xlim(-2.3, 4.8)
ax.set_ylim(-2.5, 12)
ax.legend(prop=prop, framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.35)
ax.spines[['top','right']].set_visible(False)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontproperties(prop)

# ── Panel 2: Gradient equilibrium ────────────────────────────────────────────
ax2 = axes[1]
for G, col, lab in zip(G_vals, colors, labels):
    w_star = -c / (2 * G)
    ax2.plot(w2, 2 * G * w2, color=col, linewidth=2.2, label=lab)
    ax2.plot(w_star, 2 * G * w_star, 'o', color=col, markersize=7, zorder=5)

ax2.axhline(-c, color='#555555', linewidth=2.0, linestyle='--',
            label=r'$-\nabla f^\top \mathbf{H}$')
ax2.annotate(r'$-\nabla f^\top \mathbf{H}$',
             xy=(4.35, -c + 0.08), fontproperties=prop, va='bottom', color='#555555')

ax2.set_xlabel(r'$w^k$', fontproperties=prop)
ax2.set_ylabel(r'Gradient', fontproperties=prop)
ax2.set_title(
    r"Equilibrium: $2\,\mathbf{G}_\mathrm{w}\, w^* = -\nabla f^\top \mathbf{H}$",
    fontproperties=prop14)
ax2.set_xlim(0, 4.8)
ax2.set_ylim(0, 8)
ax2.legend(prop=prop, framealpha=0.9, loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.35)
ax2.spines[['top','right']].set_visible(False)
for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
    lbl.set_fontproperties(prop)

plt.tight_layout(pad=1.8)
plt.savefig(r"C:\Users\Manuel Schwenke\Desktop\Daten\ofo_tradeoff.svg", dpi=150, bbox_inches='tight')
plt.show()
