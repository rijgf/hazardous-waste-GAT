"""Typeset the manuscript's unchanged formulas for Markdown viewers without TeX.

This is presentation-only: no experiments, solvers, models, or raw result files
are read or changed. PNG is the portable Markdown display; SVG and LaTeX retain
editable/vector versions of the same expressions.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "供应链管理写作" / "排版资源"
EQUATIONS = {
    "目标函数": (
        r"J_{\omega,i}(x)=\omega_c\frac{C_i(x)}{b_{C,i}}"
        r"+\omega_r\frac{R_i(x)}{b_{R,i}},\qquad\omega_c+\omega_r=1.",
        (9.2, 0.75),
    ),
    "弱覆盖率": (
        r"\mathcal{C}(A,B)=\frac{\left|\left\{b\in B:\ \exists a\in A,"
        r"\ C(a)\leq C(b),\ R(a)\leq R(b)\right\}\right|}{|B|}",
        (9.2, 0.88),
    ),
    "相对差异": (
        r"\Delta_J(\%)=100\,\frac{J_{\mathrm{PPO}}-J_{\mathrm{MILP}}}"
        r"{J_{\mathrm{MILP}}}",
        (9.2, 0.8),
    ),
}


def render():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"mathtext.fontset": "stix", "svg.fonttype": "path"})
    for name, (expression, size) in EQUATIONS.items():
        fig = plt.figure(figsize=size, dpi=300, facecolor="white")
        fig.text(0.5, 0.5, f"${expression}$", ha="center", va="center", fontsize=17, color="black")
        fig.savefig(DESTINATION / f"{name}.png", dpi=300, facecolor="white")
        fig.savefig(DESTINATION / f"{name}.svg", facecolor="white", metadata={"Date": None})
        plt.close(fig)
        source = "\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\pagestyle{empty}\n\\begin{document}\n\\[\n" + expression + "\n\\]\n\\end{document}\n"
        (DESTINATION / f"{name}.tex").write_text(source, encoding="utf-8")
        print(f"Rendered {name}: PNG, SVG, LaTeX")


if __name__ == "__main__":
    render()
