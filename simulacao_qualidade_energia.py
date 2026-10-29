"""Classificação de distúrbios de qualidade de energia com aprendizado de máquina (v2).

Sinais sintéticos de tensão (60 Hz) -> atributos (RMS e THD) -> classificadores
(Árvore de Decisão, SVM, MLP). Avalia desempenho em função da relação sinal-ruído (SNR),
com 10 repetições, IC 95 %, testes pareados, métricas por classe e experimentos de robustez.
Uso: python sim.py   (gera tabelas, figuras e results.json em ./out; ~10 min)
"""
import os, json, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
warnings.filterwarnings("ignore")

F0 = 60.0                # frequência fundamental (Hz)
SPC = 64                 # amostras por ciclo
FS = F0 * SPC            # frequência de amostragem (Hz)
NC = 10                  # ciclos por janela
N = SPC * NC
T = np.arange(N) / FS
CLASSES = ["Normal", "Afundamento", "Elevação", "Harmônicos"]
COMB = ["Afundamento + harmônicos", "Elevação + harmônicos"]   # só em teste
FEAT = ["RMS_min", "RMS_max", "RMS_med", "RMS_dp", "THD", "Crista"]
SEEDS = tuple(range(10))
SNRS = (None, 40, 30, 20, 10, 5)
OUT = "out"
os.makedirs(OUT, exist_ok=True)
R = {}                   # resultados numéricos para o artigo


def snr_label(s):
    return "Sem ruído" if s is None else f"{s} dB"


def add_noise(x, snr_db, rng):
    """Ruído gaussiano branco para a SNR desejada (dB), calculada sobre a potência da janela."""
    if snr_db is None:
        return x
    p_noise = np.mean(x ** 2) / (10 ** (snr_db / 10))
    return x + rng.normal(0.0, np.sqrt(p_noise), x.shape)


def harmonics(rng, v_t=T, scale=1.0):
    """Soma de harmônicas ímpares 3,5,7 (até 5 %) e 9,11 (até 2 %), fases aleatórias."""
    h_sum = np.zeros(N)
    for h in (3, 5, 7, 9, 11):
        amp = (rng.uniform(0, 0.05) if h <= 7 else rng.uniform(0, 0.02)) * scale
        h_sum += amp * np.sin(2 * np.pi * h * F0 * v_t + rng.uniform(0, 2 * np.pi))
    return h_sum


def gen_signal(cls, rng, frac=False, sag_rng=(0.50, 0.90), swell_rng=(1.10, 1.50), df=0.0):
    """Janela de tensão em pu (sem ruído).
    frac: duração/início do evento em número arbitrário de amostras (não múltiplo de ciclo).
    df: desvio da frequência real em relação a 60 Hz (Hz), para estudar vazamento espectral."""
    ph = rng.uniform(0, 2 * np.pi)
    a = rng.uniform(0.96, 1.04)
    f = F0 + df
    v = a * np.sin(2 * np.pi * f * T + ph)
    base = cls.split(" + ")[0]
    if cls == "Normal":
        h = rng.choice([3, 5, 7])
        v += rng.uniform(0, 0.015) * np.sin(2 * np.pi * h * f * T + rng.uniform(0, 2 * np.pi))
    elif base in ("Afundamento", "Elevação"):
        mag = rng.uniform(*sag_rng) if base == "Afundamento" else rng.uniform(*swell_rng)
        env = np.ones(N)
        if frac:
            dur = rng.integers(2 * SPC, N + 1)
            ini = rng.integers(0, N - dur + 1)
            env[ini:ini + dur] = mag
        else:
            dur = rng.integers(2, NC + 1)            # duração em ciclos inteiros
            ini = rng.integers(0, NC - dur + 1)
            env[ini * SPC:(ini + dur) * SPC] = mag
        v = v * env
        if cls.endswith("harmônicos"):
            v = v + harmonics(rng)
    elif cls == "Harmônicos":
        v = v + harmonics(rng)
    return v


def features(v):
    """Seis atributos: RMS por ciclo (mín., máx., médio, desvio), THD (harmônicas 2-16) e fator de crista."""
    rms_c = np.sqrt((v.reshape(NC, SPC) ** 2).mean(axis=1))
    spec = np.abs(np.fft.rfft(v)) / (N / 2)
    k1 = NC                                          # índice da fundamental (60 Hz) na janela de 10 ciclos
    fund = spec[k1]
    harm = spec[2 * k1:16 * k1 + 1:k1]               # 2ª a 16ª harmônicas
    thd = np.sqrt((harm ** 2).sum()) / fund if fund > 0 else 0.0
    crest = np.max(np.abs(v)) / np.sqrt(np.mean(v ** 2))
    return [rms_c.min(), rms_c.max(), rms_c.mean(), rms_c.std(), thd, crest]


def make_dataset(n_per_class, snr_db, rng, classes=CLASSES, **kw):
    X, y = [], []
    for ci, c in enumerate(classes):
        for _ in range(n_per_class):
            X.append(features(add_noise(gen_signal(c, rng, **kw), snr_db, rng)))
            y.append(ci)
    return np.array(X), np.array(y)


def models(seed):
    return {
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8),
                             max_iter=1500, random_state=seed)),
    }


MN = list(models(0))


def ci95(a):
    a = np.asarray(a, float)
    return stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


def plot_examples(rng):
    fig, axs = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    for ax, c in zip(axs, CLASSES):
        ax.plot(T * 1000, gen_signal(c, rng), lw=1)
        ax.set_ylabel("Tensão (pu)")
        ax.set_title(c, fontsize=9, loc="left")
        ax.grid(alpha=.3)
    axs[-1].set_xlabel("Tempo (ms)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_sinais_exemplo.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- experimento principal
def main(n_per_class=500):
    plot_examples(np.random.default_rng(42))
    rows, pcs, cm_store, leak = [], [], {}, 0
    for snr in SNRS:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * s + (0 if snr is None else snr))
            X, y = make_dataset(n_per_class, snr, rng)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
            trset = {tuple(np.round(r, 12)) for r in Xtr}
            leak += sum(tuple(np.round(r, 12)) in trset for r in Xte)      # vazamento: linhas idênticas
            for name, m in models(s).items():
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=range(4), zero_division=0)
                rows.append(dict(SNR=snr_label(snr), snr_num=999 if snr is None else snr, modelo=name,
                                 seed=s, acc=accuracy_score(yte, pred), f1_macro=f.mean()))
                for ci, c in enumerate(CLASSES):
                    pcs.append(dict(SNR=snr_label(snr), snr_num=999 if snr is None else snr, modelo=name,
                                    seed=s, classe=c, precisao=p[ci], revocacao=r[ci], f1=f[ci]))
                if s == SEEDS[0]:
                    cm_store[(snr, name)] = confusion_matrix(yte, pred, normalize="true")
    R["linhas_identicas_treino_teste"] = int(leak)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/resultados_brutos.csv", index=False)
    pc = pd.DataFrame(pcs); pc.to_csv(f"{OUT}/metricas_por_classe_brutas.csv", index=False)

    # tabela principal: média, dp e IC95 %
    tab = {}
    for (snr_n, snr, mod), g in df.groupby(["snr_num", "SNR", "modelo"]):
        tab[(snr, mod)] = dict(acc=g.acc.mean() * 100, dp=g.acc.std() * 100, ic=ci95(g.acc) * 100,
                               f1=g.f1_macro.mean() * 100, f1dp=g.f1_macro.std() * 100)
    R["acc"] = {f"{k[0]}|{k[1]}": v for k, v in tab.items()}
    pd.DataFrame([dict(SNR=k[0], modelo=k[1], **v) for k, v in tab.items()]).to_csv(
        f"{OUT}/tabela_acuracia.csv", index=False)

    # por classe (média das repetições) a 20, 10 e 5 dB
    pcm = pc.groupby(["SNR", "modelo", "classe"])[["precisao", "revocacao", "f1"]].mean() * 100
    pcm.to_csv(f"{OUT}/tabela_metricas_por_classe.csv")
    R["porclasse"] = {f"{a}|{b}|{c}": dict(P=r.precisao, R=r.revocacao, F1=r.f1)
                      for (a, b, c), r in pcm.iterrows()}

    # testes pareados entre classificadores (mesmas amostras e partições em cada repetição)
    tests = []
    for snr in SNRS:
        sub = df[df.SNR == snr_label(snr)]
        pv = {m: sub[sub.modelo == m].sort_values("seed").acc.values for m in MN}
        res = []
        for a, b in itertools.combinations(MN, 2):
            d = (pv[a] - pv[b]) * 100
            t, p = stats.ttest_rel(pv[a], pv[b])
            try:
                pw = stats.wilcoxon(pv[a], pv[b]).pvalue
            except ValueError:
                pw = 1.0
            res.append([a, b, d.mean(), ci95(d), p, pw])
        order = np.argsort([r[4] for r in res])               # correção de Holm-Bonferroni
        holm, prev = {}, 0
        for rank, i in enumerate(order):
            prev = max(prev, min(1.0, res[i][4] * (len(res) - rank)))
            holm[i] = prev
        for i, r in enumerate(res):
            tests.append(dict(SNR=snr_label(snr), A=r[0], B=r[1], dif_pp=r[2], ic_pp=r[3],
                              p_t=r[4], p_wilcoxon=r[5], p_holm=holm[i]))
    tt = pd.DataFrame(tests); tt.to_csv(f"{OUT}/testes_pareados.csv", index=False)
    R["testes"] = tests

    # Figura 2: acurácia x SNR com IC 95 %
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [snr_label(s) if s is None else f"{s}" for s in SNRS]
    labels[0] = "Sem ruído"
    for name in MN:
        ys = [tab[(snr_label(s), name)]["acc"] for s in SNRS]
        es = [tab[(snr_label(s), name)]["ic"] for s in SNRS]
        ax.errorbar(range(len(SNRS)), ys, yerr=es, marker="o", capsize=3, label=name)
    ax.set_xticks(range(len(SNRS))); ax.set_xticklabels(labels)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Acurácia (%)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_acuracia_snr.png", dpi=200); plt.close(fig)

    for snr in (20, 10):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for ax, name in zip(axs, MN):
            ConfusionMatrixDisplay(cm_store[(snr, name)], display_labels=CLASSES).plot(
                ax=ax, colorbar=False, xticks_rotation=30, values_format=".2f")
            ax.set_title(f"{name} (SNR = {snr} dB)", fontsize=10)
            ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe verdadeira")
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig{3 if snr == 20 else 4}_matriz_confusao_{snr}dB.png", dpi=200)
        plt.close(fig)
    print(pd.DataFrame([dict(SNR=k[0], modelo=k[1], acc=round(v['acc'], 1), ic=round(v['ic'], 1),
                             f1=round(v['f1'], 1)) for k, v in tab.items()]).to_string(index=False))


# ---------------------------------------------------------------- THD: demonstração
def thd_study(n=300):
    rows = {}
    for frac in (False, True):
        rng = np.random.default_rng(99)
        for c in CLASSES:
            th = np.array([features(gen_signal(c, rng, frac=frac))[4] * 100 for _ in range(n)])
            rows[f"{'frac' if frac else 'int'}|{c}"] = dict(med=np.median(th), p95=np.percentile(th, 95), mx=th.max())
    R["thd_sem_ruido"] = rows
    # vazamento espectral: desvio de frequência na classe Normal
    lk = {}
    for df_ in (0.0, 0.05, 0.1, 0.5):
        rng = np.random.default_rng(7)
        th = np.array([features(gen_signal("Normal", rng, df=df_))[4] * 100 for _ in range(n)])
        lk[str(df_)] = dict(med=np.median(th), p95=np.percentile(th, 95))
    R["thd_vazamento"] = lk
    print(json.dumps({k: {a: {b: round(c, 4) for b, c in d.items()} for a, d in v.items()}
                      for k, v in R.items() if k.startswith("thd")}, indent=1))


# ---------------------------------------------------------------- robustez
def robustez(ntr=350, nte=300):
    # (a) ruído descasado entre treino e teste
    train_conds = {"20 dB": [20], "10 dB": [10], "misto (40, 30, 20 e 10 dB)": [40, 30, 20, 10]}
    tests = (None, 40, 30, 20, 10, 5)
    acc_a = {}
    models_frac = {}
    for s in SEEDS:
        for tc, lst in train_conds.items():
            rng = np.random.default_rng(31000 + 100 * s + len(lst) * 7 + lst[0])
            parts = [make_dataset(ntr // len(lst), snr, rng) for snr in lst]
            Xtr = np.vstack([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
            ms = models(s)
            for m in ms.values():
                m.fit(Xtr, ytr)
            for te in tests:
                rng2 = np.random.default_rng(52000 + 10 * s + (0 if te is None else te))
                Xte, yte = make_dataset(nte, te, rng2)
                for name, m in ms.items():
                    acc_a.setdefault((tc, snr_label(te), name), []).append(accuracy_score(yte, m.predict(Xte)))
    R["descasado"] = {f"{k[0]}|{k[1]}|{k[2]}": dict(m=np.mean(v) * 100, ic=ci95(v) * 100) for k, v in acc_a.items()}

    for key in ("descasado",):
        print("\n==", key)
        for k, v in R[key].items():
            print(k, {a: round(b, 1) for a, b in v.items()})


def robustez_bcd(ntr=350, nte=300):
    """(b) durações fracionárias, (c) distúrbios combinados e (d) intensidade, com treino na mesma SNR do teste."""
    acc_b, comb, inten = {}, {}, {}
    bands = {"afundamento raso (0,80-0,90 pu)": ("Afundamento", dict(sag_rng=(0.80, 0.90)), 1),
             "afundamento profundo (0,50-0,60 pu)": ("Afundamento", dict(sag_rng=(0.50, 0.60)), 1),
             "elevação rasa (1,10-1,20 pu)": ("Elevação", dict(swell_rng=(1.10, 1.20)), 2),
             "elevação intensa (1,40-1,50 pu)": ("Elevação", dict(swell_rng=(1.40, 1.50)), 2)}
    for s in SEEDS:
        for te in (None, 20, 10):
            tl = snr_label(te)
            for trname, frac in (("treino com ciclos inteiros", False), ("treino com durações fracionárias", True)):
                rng = np.random.default_rng(73000 + 10 * s + frac + (0 if te is None else te))
                ms = models(s)
                Xtr, ytr = make_dataset(ntr, te, rng, frac=frac)            # treino na mesma SNR do teste
                for m in ms.values():
                    m.fit(Xtr, ytr)
                rng2 = np.random.default_rng(84000 + 10 * s + (0 if te is None else te))
                Xte, yte = make_dataset(nte, te, rng2, frac=True)          # teste com durações fracionárias
                for name, m in ms.items():
                    acc_b.setdefault((trname, tl, name), []).append(accuracy_score(yte, m.predict(Xte)))
                if not frac:                                                # modelos padrão (ciclos inteiros)
                    rng3 = np.random.default_rng(95000 + 10 * s + (0 if te is None else te))
                    for cc in COMB:
                        Xc, _ = make_dataset(nte, te, rng3, classes=[cc])
                        for name, m in ms.items():
                            comb.setdefault((cc, tl, name), []).append(
                                np.bincount(m.predict(Xc), minlength=4) / len(Xc) * 100)
                    for bn, (cc, kw, lab) in bands.items():
                        Xb, _ = make_dataset(nte, te, rng3, classes=[cc], **kw)
                        for name, m in ms.items():
                            inten.setdefault((bn, tl, name), []).append(np.mean(m.predict(Xb) == lab) * 100)
    R["fracionario"] = {f"{k[0]}|{k[1]}|{k[2]}": dict(m=np.mean(v) * 100, ic=ci95(v) * 100) for k, v in acc_b.items()}
    R["combinados"] = {f"{k[0]}|{k[1]}|{k[2]}": list(np.mean(v, axis=0)) for k, v in comb.items()}
    R["intensidade"] = {f"{k[0]}|{k[1]}|{k[2]}": dict(m=np.mean(v), ic=ci95(v)) for k, v in inten.items()}


def ablation(n_per_class=500, snrs=(20, 10)):
    sets = {"Apenas RMS": [0, 1, 2, 3], "Apenas THD": [4], "RMS mín. + RMS máx. + THD": [0, 1, 4],
            "RMS + THD + crista": [0, 1, 2, 3, 4, 5]}
    rows = []
    for snr in snrs:
        for s in SEEDS:
            rng = np.random.default_rng(777 + 1000 * s + snr)
            X, y = make_dataset(n_per_class, snr, rng)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
            for fs_name, idx in sets.items():
                m = models(s)["SVM (RBF)"].fit(Xtr[:, idx], ytr)
                rows.append(dict(SNR=f"{snr} dB", atributos=fs_name, seed=s, acc=accuracy_score(yte, m.predict(Xte[:, idx]))))
    d = pd.DataFrame(rows)
    out = {}
    for (a, b), g in d.groupby(["SNR", "atributos"]):
        out[f"{a}|{b}"] = dict(m=g.acc.mean() * 100, dp=g.acc.std() * 100, ic=ci95(g.acc) * 100)
    R["ablacao"] = out
    pd.DataFrame([dict(SNR=k.split("|")[0], atributos=k.split("|")[1], **v) for k, v in out.items()]).to_csv(
        f"{OUT}/tabela_ablacao_atributos_SVM.csv", index=False)
    print({k: round(v["m"], 1) for k, v in out.items()})


def extra_figures():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, snr in zip(axs, (None, 20, 10)):
        rng = np.random.default_rng(5 + (0 if snr is None else snr))
        Xs, ys = make_dataset(300, snr, rng)
        R.setdefault("thd_med", {})[snr_label(snr)] = {c: float(np.median(Xs[ys == i, 4] * 100)) for i, c in enumerate(CLASSES)}
        ax.boxplot([Xs[ys == c, 4] * 100 for c in range(4)], tick_labels=CLASSES, showfliers=False)
        ax.set_title(snr_label(snr) if snr is None else f"SNR = {snr} dB", fontsize=10)
        ax.tick_params(axis="x", rotation=30); ax.grid(alpha=.3)
    axs[0].set_ylabel("THD (%)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_thd_por_classe.png", dpi=200); plt.close(fig)


def importancia_atributos(n_per_class=500, snrs=(20, 10)):
    from sklearn.inspection import permutation_importance
    rows = []
    for snr in snrs:
        for s in SEEDS:
            rng = np.random.default_rng(2000 * s + snr)
            X, y = make_dataset(n_per_class, snr, rng)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
            m = DecisionTreeClassifier(max_depth=8, random_state=s).fit(Xtr, ytr)
            pi = permutation_importance(m, Xte, yte, n_repeats=10, random_state=s)
            for i, f in enumerate(FEAT):
                rows.append(dict(SNR=f"{snr} dB", atributo=f, gini=m.feature_importances_[i], perm=pi.importances_mean[i]))
    d = pd.DataFrame(rows)
    g = d.groupby(["SNR", "atributo"]).agg(gini=("gini", "mean"), gini_dp=("gini", "std"),
                                           perm=("perm", "mean"), perm_dp=("perm", "std")).reset_index()
    g.to_csv(f"{OUT}/tabela_importancia_atributos_arvore.csv", index=False)
    R["importancia"] = {f"{r.SNR}|{r.atributo}": dict(gini=r.gini * 100, perm=r.perm * 100) for r in g.itertuples()}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, snr in zip(axs, ("20 dB", "10 dB")):
        sub = g[g.SNR == snr].set_index("atributo").loc[FEAT]
        ax.barh(FEAT[::-1], (sub["gini"] * 100)[::-1], xerr=(sub["gini_dp"] * 100)[::-1], capsize=3)
        ax.set_title(f"SNR = {snr}", fontsize=10); ax.set_xlabel("Importância (%)"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_importancia_atributos.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["main", "thd", "robustez", "robustez_bcd", "ablacao", "extra", "imp"]
    fn = dict(main=main, thd=thd_study, robustez=robustez, robustez_bcd=robustez_bcd, ablacao=ablation, extra=extra_figures, imp=importancia_atributos)
    for p in parts:
        fn[p]()
        old = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
        old.update(R)
        json.dump(old, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
