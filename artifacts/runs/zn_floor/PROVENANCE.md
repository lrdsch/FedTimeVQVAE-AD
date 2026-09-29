# zn_floor — provenienza delle righe

**Le 9 serie qui sotto non sono state ricalcolate il 2026-08-06: sono state importate da
`evidence/records_pre_znorm_20260804_floor.json` con `scripts/zn_floor_import.py`.**

I numeri sono gli stessi che produrrebbe una run di oggi — verificato, non assunto. Ma
questa riga esiste perché in questo repo i risultati si sono già persi due volte per
provenienza confusa (il cutoff z-norm del 04/08, la purga totale del 30/07), e una sessione
futura che si chiede «quando l'abbiamo girato?» deve poter rispondere.

## Perché il cutoff z-norm non tocca il floor

Il floor **non applica mai** `window_normalization`. In `floor_eval.accumulate` la
`SlidingWindowDataset` serve solo per `.indices`, cioè i confini delle finestre; i valori
arrivano da `FH._windows(x, W, blk)` sulla serie grezza. La z-norm vive in
`SlidingWindowDataset.__getitem__` (`data.py:384`), che il floor non chiama.

Il cutoff del 2026-08-04 invalida le run **deep**, dove la z-norm cambia il modello. Sul
floor è un no-op, quindi il nome del file d'archivio è fuorviante: quelle righe non sono
«pre-znorm», sono semplicemente il floor.

Non è un problema di equità: la z-norm per finestra **acceca il deep** su livello assoluto e
ampiezza, che il floor grezzo invece vede. Il floor non normalizzato è quindi la baseline
**più informata**, e batterlo è un risultato *a fortiori*.

## I sei controlli, tutti passati prima di scrivere

| # | controllo | esito |
|---|---|---|
| 1 | il codice non è cambiato | `floor_heads.py` 30/07 · `floor_eval.py` 01/08 · git pulito |
| 2 | lo schema combacia | record `_schema=2` == `floor_eval.SCHEMA` di oggi |
| 3 | il build non è cambiato | `data/raw/ucr_split_w2p/` fermo al 29/07, nulla dopo il 04/08 |
| 4 | stessa griglia di valutazione | floor stride 41·18·33·41·97·54·67·20·25 == `eval_stride_rate=0.1 × W` del percorso deep, serie per serie |
| 5 | **riproduzione bit-identica** | rilanciato `ucr_001` col codice di oggi: **360 metriche su 3 arm × 5 client, max\|Δ\| = 0.000e+00** (`ma_c` k=10 — la riga del paper — `ma_causal`, `pca`; `ar` non è arrivato in fondo entro il timeout della sonda) |
| 6 | i doppioni non sono misure diverse | 63 arm compaiono due volte (`ucrNNN_floor` / `_floor100`): **6615 metriche, max\|Δ\| = 0.000e+00**; l'unico scarto è `_secs` |

## Il dedup, e perché non era innocuo

Su `ucr_001`, `ucr_011` e `ucr_222` ogni arm compare due volte. Le due varianti sono la
stessa misura (controllo 6); `floor100` porta **in più** la colonna
`paper_top1_acc_at_100` — la tolleranza ±100 del protocollo UCR, calcolata offline perché
`metrics_tolerance` sta nel fingerprint della coorte e non si tocca.

⚠️ **Il nome del file non contiene la tolleranza, quindi i due collidono.** Un «primo che
arriva vince» tiene la variante povera su metà delle serie — è esattamente l'errore che ho
fatto al primo import e che ho dovuto rifare. `zn_floor_import.py` ora sceglie
esplicitamente il record con più campi non-`None`.

## Copertura

189 arm · 945 righe · 9 serie su 10.

Teste: `ma_c` k=10 (**la riga del paper**, zero parametri) · `ma_causal` k=10 · `ar` p=32
λ=1e-4 · `pca` K=8 γ=0.05. Le prime due sono stampate `arm_invariant` e non `local`: per una
testa a zero parametri local, central e ogni modo federato sono bit-identici per costruzione.

⚠️ **`ucr_082` non c'è in archivio.** È l'unica serie lanciata davvero, il 2026-08-06.

⚠️ `threshold_rule = quantile_fallback`: solo le metriche **threshold-free** (AUROC, AUPRC,
VUS-PR, top-K) sono confrontabili con gli arm deep. Le F1 no.

## Il risultato, al 2026-08-06

Mediana sui 5 client, `ma_c` k=10 contro il deep — battono il floor:

| | VUS-PR | AUPRC |
|---|---|---|
| `centralized` | 9/9 | 9/9 |
| **A2** | **8/8** | **8/8** |
| `local` | 8/9 | 8/9 |

`local` perde su `ucr_043` (0.022 contro 0.047 del floor su VUS-PR). Il floor batte quindi
il puro locale su una serie: va riportato, non nascosto.
