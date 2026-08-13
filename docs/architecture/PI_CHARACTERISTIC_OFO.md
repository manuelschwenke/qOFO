# PI-Regelcharakteristik des DSO-OFO-Controllers

> **Status (2026-07-31): historisch.** Der hier beschriebene I-Anteil
> (`g_qi`, `lambda_qi`, `q_integral_max_mvar`, `_q_error_integral`,
> `reset_integral()`) wurde aus `controller/dso_controller.py` und allen
> Config-Dataclasses entfernt. Er war in allen Thesis-Konfigurationen
> deaktiviert (`g_qi = 0`). Das Dokument bleibt als Herleitung der
> P-/PI-Charakteristik erhalten; die Abschnitte zum I-Anteil beschreiben
> keinen im Code vorhandenen Mechanismus mehr.

## Regelstrecke und Stellglieder

Der DSO-OFO-Controller regelt die Blindleistung an der ÜN-VN-Schnittstelle
$Q_\text{if}$ auf einen vom ÜN-Controller vorgegebenen Sollwert $Q_\text{set}$.

Stellgrößenvektor:

$$
u = \begin{pmatrix} Q_\text{DER} \\ s_\text{OLTC} \\ \text{state}_\text{shunt} \end{pmatrix}
\quad \in \mathbb{R}^{n_u}
$$

mit kontinuierlichen (EZA-Blindleistung) und diskreten (Stufensteller, Shunts)
Stellgliedern.

## Klassischer OFO-Regelkreis (P-Anteil)

Die Standard-OFO-Iteration lautet:

$$
u^{k+1} = u^k + \alpha \cdot \sigma^k
$$

wobei $\sigma^k$ die Lösung des MIQP ist:

$$
\sigma^k = \arg\min_w \; g(w, z)
\quad \text{s.t.} \quad \text{Ein-/Ausgangsbeschränkungen}
$$

Der Zielfunktionsgradient für das Q-Tracking ist:

$$
\nabla f_Q^k = 2 \, g_Q \, \bigl(Q_\text{if}^k - Q_\text{set}\bigr)^T
\frac{\partial Q_\text{if}}{\partial u}
$$

Dies entspricht einer **proportionalen Rückführung** des Regelfehlers
$e_Q^k = Q_\text{if}^k - Q_\text{set}$ über die Sensitivitätsmatrix in den
Stellgrößenraum. Der Regelkreis verhält sich wie ein P-Regler mit
effektiver Verstärkung $\alpha \cdot g_Q / g_w$.

### Stationäre Genauigkeit

Im stationären Zustand ($\sigma^k = 0$) gilt
$\nabla f_Q = 0$ nur dann, wenn $e_Q = 0$, **sofern keine
Regularisierung** ($g_u = 0$) vorliegt. Ist $g_u > 0$, entsteht eine
bleibende Regelabweichung (Bias), da die Regularisierung das Optimum
von der perfekten Sollwerterfüllung weg verschiebt.

## Erweiterung um I-Anteil (Leaky Integrator)

### Motivation

Bei großen Sollwertabweichungen, die die kontinuierlichen Stellglieder
(EZA) allein nicht ausgleichen können, reicht der P-Gradient häufig
nicht aus, um die hohe Änderungsstrafe $g_w$ der diskreten Stellglieder
(OLTC, Shunts) zu überwinden. Der Fehler bleibt bestehen, obwohl eine
diskrete Schalthandlung ihn beseitigen könnte.

### Leaky-Integrator-Formulierung

Der Integralzustand wird als **exponentiell gewichtete Fehlersumme**
(Leaky Integrator) geführt:

$$
s^{k+1} = \lambda \cdot s^k + e_Q^k
$$

mit dem Decay-Faktor $\lambda \in [0, 1]$:
- $\lambda = 1$: reine Integration (kein Vergessen)
- $\lambda < 1$: exponentielle Abschwächung vergangener Fehler

Der erweiterte Zielfunktionsgradient ist:

$$
\nabla f^k = \underbrace{2 \, g_Q \, (e_Q^k)^T \frac{\partial Q_\text{if}}{\partial u}}_{\text{P-Anteil}}
+ \underbrace{2 \, g_{Q,I} \, (s^k)^T \frac{\partial Q_\text{if}}{\partial u}}_{\text{I-Anteil}}
$$

### Analogie zum zeitdiskreten PI-Regler

Ein klassischer zeitdiskreter PI-Regler mit Abtastzeit $T_s$ hat die Form:

$$
u^k = K_P \, e^k + K_I \, T_s \sum_{j=0}^{k} e^j
$$

Im OFO-Framework entsteht die PI-Struktur implizit:

| Klassischer PI | OFO-PI |
|----------------|--------|
| $K_P$ | $\alpha \cdot g_Q / g_w$ |
| $K_I \cdot T_s$ | $\alpha \cdot g_{Q,I} / g_w$ |
| $\sum e^j$ (reine Integration) | $s^k = \lambda \, s^{k-1} + e_Q^k$ (Leaky Integrator) |
| Anti-Windup (Clamp) | $\|s^k\|_\infty \leq s_\text{max}$ |

Der wesentliche Unterschied: Im OFO wirkt der PI-Gradient nicht direkt
als Stellgrößenänderung, sondern als Kostenfunktionsgradient innerhalb
des MIQP. Das MIQP berücksichtigt gleichzeitig Beschränkungen
(Spannungsbänder, thermische Limits, Stellbereichsgrenzen), was eine
constraint-konforme PI-Regelung ermöglicht.

### Geschlossener Ausdruck für den Integralzustand

Durch rekursive Einsetzung erhält man:

$$
s^k = \sum_{j=0}^{k} \lambda^{k-j} \, e_Q^j
$$

Für $\lambda < 1$ konvergiert die geometrische Reihe, und der stationäre
Integralwert bei konstantem Fehler $\bar{e}$ beträgt:

$$
\bar{s} = \frac{\bar{e}}{1 - \lambda}
$$

Dies bedeutet: Kleinere $\lambda$-Werte begrenzen den maximalen
Integraldruck auch ohne explizites Anti-Windup, auf Kosten einer
geringeren Fähigkeit zur Eliminierung bleibender Regelabweichungen.

### Anti-Windup

Zur Begrenzung des Integralzustands wird ein elementweiser Clamp
angewendet:

$$
s_i^k \leftarrow \text{clip}\bigl(s_i^k, -s_\text{max}, +s_\text{max}\bigr)
\quad \forall \, i = 1, \ldots, n_\text{if}
$$

Dies verhindert übermäßigen Integralaufbau, wenn die DSO-Stellglieder
an ihren Kapazitätsgrenzen arbeiten (z.B. alle EZA an $Q_\text{max}$,
keine weiteren OLTC-Stufen verfügbar).

## Parameter und Einstellempfehlungen

| Parameter | Symbol | Typ | Empfehlung |
|-----------|--------|-----|------------|
| `g_qi` | $g_{Q,I}$ | Integralverstärkung | $0.05 \ldots 0.2 \cdot g_Q$ |
| `lambda_qi` | $\lambda$ | Decay-Faktor | $0.8 \ldots 1.0$ |
| `q_integral_max_mvar` | $s_\text{max}$ | Anti-Windup-Grenze | Typisch 20–100 Mvar |

### Einstellstrategie

1. **Start konservativ**: $g_{Q,I} = 0.05 \cdot g_Q$, $\lambda = 0.9$
2. **Integraldruck erhöhen**: $g_{Q,I}$ schrittweise erhöhen, bis
   persistente Fehler zur Schalthandlung führen
3. **Decay anpassen**: $\lambda \to 1.0$ für stärkeren Druck,
   $\lambda \to 0.8$ bei Oszillationsneigung
4. **Anti-Windup**: $s_\text{max}$ so wählen, dass der Integralterm
   die Änderungsstrafe $g_w$ der größten diskreten Schalthandlung
   überwinden kann

## Blockschaltbild

```
                              ┌──────────────────┐
         Q_set ──────────(-)──┤  e_Q^k           │
                          │   │                   │
         Q_if^k ──────────┘   │  ┌─────────────┐ │
                              │  │ P:           │ │
                              ├──│ 2·g_Q·e^T·H  │─┤
                              │  └─────────────┘ │
                              │                   │       ┌──────────┐      ┌──────────┐
                              │  ┌─────────────┐ │       │          │      │          │
                              │  │ I (Leaky):  │ │       │          │      │          │
                              ├──│ 2·g_QI·s^T·H│─┼─ ∇f ──│  MIQP    │─ σ ──│  u + ασ  │── u^{k+1}
                              │  │             │ │       │          │      │          │
                              │  │ s=λs+e      │ │       │  s.t.    │      │          │
                              │  │ clip(s,±max)│ │       │  limits  │      │          │
                              │  └─────────────┘ │       └──────────┘      └──────────┘
                              └──────────────────┘
                               Gradient-Berechnung           Solver          OFO-Update
```

## Implementierung

Datei: `controller/dso_controller.py`

- Konfiguration: `DSOControllerConfig.g_qi`, `.lambda_qi`, `.q_integral_max_mvar`
- Zustand: `DSOController._q_error_integral` (Vektor, $n_\text{interfaces}$ Elemente)
- Gradient: In `_compute_objective_gradient()`, nach dem P-Anteil
- Reset: `DSOController.reset_integral()` für manuelles Zurücksetzen
- Deaktivierung: `g_qi = 0.0` (Standard) schaltet den I-Anteil vollständig ab
