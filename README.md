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

```bash
git clone https://github.com/example/cognisim && cd cognisim
make check     # what is installed, what is missing
make code      # run the prototype: 4 seconds, 9 CSV files
make           # build all four artifacts
```

---

## What is here

| Folder | What it builds | How long |
|---|---|---|
| [`code/`](code) | `cognisim.py`, the reference prototype: sensing, load inference, knowledge tracing, the adaptation policy, federated learning with differential privacy, and a gradient inversion attack | 4 s |
| [`paper/`](paper) | the 3 page AAAI paper, 2 pages of body plus 1 of references | 15 s |
| [`slides/`](slides) | a 14 slide animated PowerPoint deck and the 5 minute talk script | 20 s |
| [`video/`](video) | the 4:10 narrated demonstration video, with burned in captions | 3 min |
| [`figures/`](figures) | the nine source figures, one copy, used by all four | — |

Every folder has its own README explaining how that part works and what to
edit. [`docs/REPRODUCE.md`](docs/REPRODUCE.md) walks from a clean machine to
all four artifacts in one pass.

---

## What the prototype shows

`code/cognisim.py` is a single file with one dependency and no arguments. It
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

## How the four parts fit together

```
figures/                    nine diagrams, the single source
   |
   +--> paper/build.sh      copies them in, runs pdflatex and bibtex     -> main.pdf
   |
   +--> slides/make_deck.js draws 14 slides with pptxgenjs               -> .pptx (no animation)
   |         add_animations.py writes the <p:timing> tree PowerPoint needs
   |                                                                     -> CogniSim_talk.pptx
   +--> video/slides.py     renders the same story in headless chromium,
             |              15 entrance frames per slide
             build_video.py speaks the script, lays out the timeline from
                            the measured audio, burns the captions        -> .mp4

code/cognisim.py            produces every number the other three quote   -> out/*.csv
```

Two design decisions are worth knowing before you edit anything.

**The narration decides the video's timing, not a hand written storyboard.**
`build_video.py` synthesises each line, measures it, and holds its slide for
exactly that long. Change one sentence and everything after it retimes itself,
captions included.

**PowerPoint animation cannot be written by any generator library.** It lives
in a separate `<p:timing>` tree at the end of each slide's XML. So the deck is
built in two steps: `make_deck.js` draws the shapes and names them, then
`add_animations.py` reads those names and writes the timing tree itself. A
shape called `a2_03` appears on click 2, third in that cascade. To move an
element to a different click, rename it.

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
| `paper` | a TeX distribution with `newtx` and `tex-gyre` (`texlive-latex-recommended texlive-fonts-extra texlive-plain-generic tex-gyre`) |
| `slides` | `node`, then `npm install pptxgenjs` |
| `video` | `espeak-ng`, `ffmpeg`, the Carlito font, then `pip install playwright pillow && playwright install chromium` |

`make check` tells you what is missing without building anything.

---

## Reusing parts of this

The pieces are deliberately separable:

- **`code/cognisim.py`** is self contained. Copy it out, change `SEED`, or
  import its classes (`SoftmaxClassifier`, `BKT`, `TrainingEnv`,
  `federated_train`, `inversion_attack`) into your own experiment.
- **`slides/add_animations.py`** works on any deck, not just this one. Name
  your shapes `a<click>_<order>` in whatever tool you use and run it.
- **`video/build_video.py`** is a general narration driven video builder. Swap
  the `SCRIPT` list and the slide renderer and it will retime itself.

---

## Publishing this repository

Nothing here is committed for you, so the first push is yours to make:

```bash
git init
git add -A
git commit -m "CogniSim: prototype, paper, deck and video"
git remote add origin git@github.com:<you>/cognisim.git
git push -u origin main
```

`.gitignore` keeps the build products out, including the three large binaries:
`slides/CogniSim_talk.pptx`, `video/CogniSim_AAAI27_demo.mp4` and
`paper/main.pdf`. Attach those to a GitHub release rather than committing them,
so the history stays small and people still get one click downloads:

```bash
gh release create v1.0 \
    slides/CogniSim_talk.pptx \
    video/CogniSim_AAAI27_demo.mp4 \
    paper/main.pdf \
    --title "AAAI-27 demonstration submission" \
    --notes "Paper, animated deck, narrated video, and the prototype that produces the numbers in all three."
```

`paper/main.bbl` is committed on purpose, so the paper compiles for someone who
does not have bibtex set up.

---

## Citing

See [`CITATION.cff`](CITATION.cff), or cite the demonstration paper once it has
a DOI. The paper's own bibliography is in `paper/references.bib`.

## Licence

MIT, see [`LICENSE`](LICENSE). The figures are the authors' own work and are
covered by the same licence.

Contact: atik.mahabub@inrs.ca
