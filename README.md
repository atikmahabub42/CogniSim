# CogniSim

**A simulator trainer that reads cognitive load, and keeps it private.**

When a pilot makes a mistake in a flight simulator, a normal training system
learns one thing: the answer was wrong. It cannot tell whether the pilot was
overloaded or simply did not know what to do, and those two problems need
opposite fixes. CogniSim watches the pilot instead. It reads eye movement,
pupil size, heart rhythm and brain activity, works out in under a second how
loaded the pilot is, and picks the next change to the lesson from that reading
together with what the pilot has already mastered. A virtual first officer
speaks the change inside the scenario, so the flight never stops. The sensor
data stays on the training computer, and only blurred summaries are shared.

This repository holds everything behind the AAAI-27 demonstration submission:
the runnable prototype, the paper source, the talk deck and the video, plus the
scripts that rebuild each one from nothing.

---

## What the prototype shows

`cognisim.py` is a single file with one dependency and no arguments. It
runs every stage of the system end to end and writes nine CSV files. Two runs
at the same seed produce byte identical output, so any number below can be
regenerated and checked.

| Quantity | Value |
|---|---|
| Load classification accuracy / macro F1 | 0.849 / 0.810 |
| Detection latency, event to decision | 0.48 s against a 0.70 s budget |
| Time to competency, fixed syllabus → adaptive | 55.3 → 47.6 min (−14 %) |
| Critical errors per trainee | 2.54 → 1.03 (−59 %) |
| Federated accuracy, no protection → ε<sub>total</sub> = 3 | 0.861 → 0.840 |
| Gradient inversion, protection off → on (ε = 2) | cosine 1.000 → 0.338 |

### What is real and what is simulated

**No human data is used or required.** The physiological streams come from a
generative sensor model whose parameter ranges are taken from the literature
cited in the paper. They are plausible synthetic signals, not recordings.

Everything downstream of the sensors is a real implementation working on those
signals: baseline corrected feature extraction, a class weighted softmax load
classifier, Bayesian knowledge tracing whose learning rate follows the germane
load estimate, tabular Q learning against a fixed hours based syllabus, FedAvg
with clipping and two tier Gaussian differential privacy, and an analytic
gradient inversion attack.

So the numbers above are properties of the pipeline **under a simulator**.
They are not results from human trainees, and neither the paper nor the talk
presents them as such. Two of them deserve a caveat in public:

- The **14 % cut in time to competency** is well short of the 28 to 34 % the
  original manuscript projects from published effect sizes. The simulator's
  learning dynamics are deliberately conservative. Treat the larger figure as
  the hypothesis a trial would test, not as something this code confirms.
- The **steady load class is the weak one** (F1 0.60, against 0.92 for the
  acute spike). Steady task load really is harder to separate from baseline
  than a sudden one, and the same asymmetry appears in the literature.

The privacy result is the one that transfers directly. The attack recovers the
input almost exactly with the protection off, and gets noise with it on, for
about two points of accuracy.

---


## Installing

Debian or Ubuntu, everything at once:

```bash
make deps
```

Or by hand, depending on which parts you want:

| To build | You need |
|---|---|
| `code` | `pip install numpy` |


`make check` tells you what is missing without building anything.

---

## Reusing parts of this

The pieces are deliberately separable:

- **`cognisim.py`** is self contained. Copy it out, change `SEED`, or
  import its classes (`SoftmaxClassifier`, `BKT`, `TrainingEnv`,
  `federated_train`, `inversion_attack`) into your own experiment.


---

## Citing

See [`CITATION.cff`](CITATION.cff), or cite the demonstration paper once it has
a DOI. The paper's own bibliography is in `paper/references.bib`.

## Licence

MIT, see [`LICENSE`](LICENSE). The figures are the authors' own work and are
covered by the same licence.

Contact: atik.mahabub@inrs.ca
