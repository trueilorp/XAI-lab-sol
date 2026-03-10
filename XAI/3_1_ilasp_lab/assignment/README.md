# 🎓 ILASP Lab - RockSample Exercise

## 📖 Descrizione dell'Esercizio

Questo esercizio utilizza **ILASP** (Inductive Learning of Answer Set Programs) per apprendere regole logiche che determinano quali rocce sono "buone" da campionare nel problema RockSample.

### Dati Disponibili

Per ogni roccia hai due informazioni:
- **`dist(Rock, Distance)`**: Distanza Manhattan dalla roccia (0-24)
- **`guess(Rock, Probability)`**: Probabilità stimata che la roccia sia buona (0-100, a passi di 10)

### Obiettivo

Apprendere regole del tipo:
```prolog
good(R) :- guess(R, G), G >= 80.
good(R) :- guess(R, G), G >= 70, dist(R, D), D <= 10.
```

## 🗂️ File del Progetto

- **`ilasp_task.las`**: File principale con mode bias ed esempi
- **`clean_rules.py`**: Script Python per filtrare regole inutili
- **`run_ilasp.sh`**: Script bash che automatizza l'intero workflow
- **`rocksample_ilp.lp`**: File dove inserire l'ipotesi finale appresa

## 🚀 Workflow Completo

### Metodo 1: Automatico (Raccomandato)

```bash
# Rendi eseguibile lo script
chmod +x run_ilasp.sh

# Esegui tutto il workflow
./run_ilasp.sh
```

Questo script:
1. ✅ Genera lo search space
2. 🧹 Pulisce le regole inutili
3. 🔍 Esegue il discovery
4. 📊 Mostra i risultati

### Metodo 2: Manuale (Passo per Passo)

#### **Passo 1: Genera lo Search Space**

```bash
ILASP --version=4 --max-rule-length=6 -ml=4 -nc -s ilasp_task.las > s_m.txt
```

**Cosa fa:**
- Genera tutte le possibili regole basate sul mode bias
- Parametri:
  - `--max-rule-length=6`: massimo 6 letterali per regola
  - `-ml=4`: massimo 4 letterali nel body
  - `-nc`: nessun constraint
  - `-s`: search space mode

**Output:** `s_m.txt` con centinaia/migliaia di regole

#### **Passo 2: Pulisci le Regole**

```bash
python3 clean_rules.py s_m.txt s_m_clean.txt
```

**Cosa fa:**
- Rimuove regole troppo generiche (senza confronti)
- Rimuove regole non informative
- Mantiene solo regole con operatori aritmetici (>=, <=, etc.)

**Output:** `s_m_clean.txt` con regole filtrate

#### **Passo 3: Prepara il File per Discovery**

```bash
# Inserisci le regole pulite nel file originale
# Questo può essere fatto manualmente o con uno script
```

Devi creare un file che contiene:
1. Background knowledge
2. Mode bias
3. **Search space pulito** (da s_m_clean.txt)
4. Esempi

#### **Passo 4: Esegui Discovery**

```bash
ILASP --version=4 -d ilasp_task_with_sm.las > ilasp_results.txt
```

**Cosa fa:**
- Cerca l'ipotesi migliore tra le regole candidate
- `-d`: discovery mode
- Può richiedere molto tempo (minuti o ore)

**Output:** `ilasp_results.txt` con le ipotesi apprese

#### **Passo 5: Analizza i Risultati**

Apri `ilasp_results.txt` e cerca:

```
Hypothesis 1 (score: 0.95, counterexamples: 5):
good(V0) :- guess(V0,V1), V1>=80.

Hypothesis 2 (score: 0.92, counterexamples: 12):
good(V0) :- guess(V0,V1), V1>=70, dist(V0,V2), V2<=10.
```

**Scegli l'ipotesi con:**
- ✅ Meno counterexamples
- ✅ Score più alto
- ✅ Regole più interpretabili

## 📊 Spiegazione del Mode Bias

### Mode Head (cosa può stare nella testa)

```prolog
#modeh(1, good(var(rock))).
```

Dice: "Posso generare regole con `good(R)` nella testa"

### Mode Body (cosa può stare nel corpo)

```prolog
#modeb(1, guess(var(rock), var(guess_value))).
#modeb(1, dist(var(rock), var(ranges_dist))).
```

Dice: "Posso usare `guess` e `dist` nel corpo delle regole"

### Costanti per Confronti

```prolog
#constant(const_value, 50).
#constant(const_value, 60).
#constant(const_value, 70).
#constant(const_value, 80).
#constant(const_value, 90).
```

Definisce le soglie da usare nei confronti.

### Operatori Aritmetici

```prolog
#modeb(1, var(guess_value) >= const(const_value)).
#modeb(1, var(guess_value) > const(const_value)).
#modeb(1, var(guess_value) <= const(const_value)).
#modeb(1, var(guess_value) < const(const_value)).
```

Permette regole con confronti tipo: `G >= 80`, `D <= 10`, ecc.

## 🔍 Analisi degli Esempi

Gli esempi hanno la forma:

```prolog
#pos(ex100043, {}, {good(_)}, {
    dist(0,14). dist(1,18). dist(2,10). dist(3,19). 
    guess(0,10). guess(1,40). guess(2,90). guess(3,40).
}).
```

**Interpretazione:**
- **`{}`**: inclusioni (cosa DEVE essere vero)
- **`{good(_)}`**: esclusioni (almeno una roccia DEVE essere good)
- **Context**: distanze e probabilità per ogni roccia

In questo esempio:
- Roccia 2: `guess=90`, `dist=10` → probabilmente questa è good!
- Rocce 0,1,3: `guess=10-40` → probabilmente non good

## 🎯 Tips per Buoni Risultati

1. **Mode bias bilanciato**: 
   - Non troppo restrittivo (poche regole)
   - Non troppo generale (troppe regole inutili)

2. **Soglie ragionevoli**:
   - Per guess: 50, 60, 70, 80, 90
   - Per dist: 5, 10, 15

3. **Pulizia aggressiva**:
   - Rimuovi regole senza confronti
   - Mantieni solo regole discriminative

4. **Controllo risultati**:
   - Verifica che le regole abbiano senso
   - Testa su alcuni esempi manualmente

## 🐛 Troubleshooting

### ILASP non trovato
```bash
# Installa ILASP da:
# https://github.com/ilaspltd/ILASP-releases
```

### Discovery troppo lento
```bash
# Riduci il search space:
# - Usa soglie più selective
# - Riduci max-rule-length
# - Pulisci più aggressivamente le regole
```

### Troppe counterexamples
```bash
# Il mode bias potrebbe essere troppo restrittivo
# Aggiungi più soglie o operatori
```

## 📚 Risorse

- [ILASP Documentation](https://ilasp.com/)
- [Answer Set Programming](https://en.wikipedia.org/wiki/Answer_set_programming)
- [RockSample Problem](https://www.ijcai.org/Proceedings/07/Papers/187.pdf)

## ✅ Checklist Completamento

- [x] Mode bias completato in `ilasp_task.las`
- [ ] Search space generato (`s_m.txt`)
- [ ] Regole pulite (`s_m_clean.txt`)
- [ ] Discovery eseguito (`ilasp_results.txt`)
- [ ] Ipotesi migliore selezionata
- [ ] Regole inserite in `rocksample_ilp.lp`
- [ ] Testing dell'ipotesi finale

## 🎉 Congratulazioni!

Una volta completato, avrai appreso automaticamente regole logiche per il problema RockSample usando tecniche di Inductive Logic Programming!
