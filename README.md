# Veil

Can this model produce this person, and if it can, what closes that off? Give Veil a link to a foundation model and photos of a person who asked to be protected. It runs **find → block → verify** and hands you two things:

- a **guard**: a standard LoRA `.safetensors` any host can load;
- a **report** measuring what the guard suppresses, how much it costs an attacker to get past, and how little it disturbs everyone else.

Veil is the protective counterpart to [Scorpion](https://github.com/riteshpakala/Scorpion):
- Scorpion asks whether an image was *trained in*. That is evidence.
- Veil asks whether a person is *reachable*, and blocks the routes that reach them. That is protection.

**What a guard is, and isn't.**
- **It blocks routes, not the likeness itself.** A guard closes the routes Veil searched: the person's names, the descriptions you give, and the prompts search finds. It holds only up to a stated attack budget. Textual-inversion-style attacks recover erased concepts given enough freedom (Pham et al., ICLR 2024), so no report says "cannot generate".
- **It only binds where the deployer applies it:** generation APIs, hosting platforms, apps. On open weights, anyone can remove it or fine-tune the person back in.

That is still a large, real market. It is also where a protected person has leverage: asking a platform to apply this adapter.

| Product | What it is |
|---|---|
| `VeilKit` | Model-agnostic core: link resolution, photos and likeness weighting, pull measurement, guard fitting and training, verification, export, reports, and the analytic toy world |
| `VeilFlux2` | The first executor: FLUX.2 Klein (klein-base-4B validated; the distilled 4B and 9B load from their configs) through Frigate's FluxKit |
| `veil` | Command-line tool |
| `VeilApp` | macOS app, launched from `main.swift` via `NSApplication` |

## Requirements

- macOS 14+ on Apple Silicon
- Swift 6.3+ (Xcode 26) with the Metal toolchain (`xcrun metal` must work)
- [Frigate](https://github.com/rao-studios/Frigate) as a sibling checkout (`../Frigate`) on branch `scorpion-fluxkit-hooks`, which provides:
  - the FluxKit linear hook and the in-memory stores;
  - the VAE encoder;
  - the text encoder built from a store, and its input-embedding path.

  Until that branch is published, `Package.swift` uses the path dependency. After that, swap it for a URL and revision.

MLX loads a small precompiled Metal library that `swift build` can't produce, so build it once after each build:

```sh
swift build
scripts/build-metallib.sh            # builds mlx.metallib and places it next to the binaries
```

For tests:
```sh
swift build --build-tests && scripts/build-metallib.sh && swift test --skip-build --no-parallel
```
- Run the suites serially: MLX's evaluation and compile locks can deadlock when two threads evaluate models at once. For the same reason, one process runs one job at a time.
- The Klein tests run when the weights are on disk and skip otherwise. They use klein-base-4B's transformer plus the `mlx-community/flux2-klein-4b-4bit` export.

## Usage

```sh
# What is this model? No weights are fetched.
veil inspect --model https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B

# Which routes reach the person? No guard is fitted.
veil assess --model https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B \
    --photos ~/veil/person --name "Jane Doe" --describe "the lead singer of …" \
    --controls ~/veil/controls --consent self --json assess.json --figure pull.png

# The whole thing: assess, fit, export, and verify the exported file.
veil guard --model … --photos ~/veil/person --name "Jane Doe" --controls ~/veil/controls \
    --consent self --out jane-guard.safetensors --json report.json --figure pull.png

# Just the guard: no assessment. One person, or a folder with one subfolder per person.
veil protect --model … --people ~/veil/people --controls ~/veil/controls \
    --consent representative --out group-guard.safetensors --json report.json

# Verify any guard (a path or a link) on a model, with fresh draws.
veil verify --model … --photos … --name "Jane Doe" --controls … --consent self --guard jane-guard.safetensors

# Custody: delete everything Veil holds for a subject.
veil forget <subject-id>        # or: veil forget --list

# Known-answer checks.
veil research toy                                            # analytic world, go/no-go
veil research sampled --model <klein folder or link> --name "Abraham Lincoln" \
    --control "Ulysses S. Grant" --control "Frederick Douglass"   # the model's own samples as the subject
```

**Run options**

| Option | Meaning |
|---|---|
| `--photos` | Photos of the person: files or folders, repeatable. 5–20 from different days, light and angles. A single photo works, but its held-out set is augmentations of it (flagged). |
| `--name` | A name or alias, repeatable; the first is the primary name. |
| `--describe` | A description that might reach the person without naming them. Repeatable. |
| `--controls DIR` | Photos of other people. A subfolder's name is that person's identity (`controls/Ulysses S. Grant/*.jpg`), which adds a "their own name still works" check. Include look-alikes. |
| `--consent` | Required. `self` (you are the person) or `representative` (you are authorized to act for them). |
| `--profile` | `quick` \| `standard` \| `thorough` (`toy` for the toy world) |
| `--anchor` | What a name is replaced with (default "a person"). |
| `--search` | Discrete prompt search for routes that don't name the person. What it finds is sealed. |
| `--with-adapter` | An adapter that is part of the deployment you are protecting (base + community LoRA). |
| `--reveal-routes` / `--redact-names` | Put discovered prompts into the report, or hash the names in it. |

**Guard options:** `--method both|closed-form|trained`, `--format diffusers|bfl`, `--out`, `--force-guard`, `--preserve "<prompt>"`.

**Protect options:** `--people DIR` (repeatable), `--no-check`, `--full-verify`, `--train-steps`, `--templates`, plus the guard options above.

### Protecting a group

`veil protect` skips the assessment and fits one guard for one person or many. Use it when the
deployer has already decided these people are not to be generated: whether the base model can
currently reach them is a different question, and it is the expensive one.

```
~/veil/people/
  Abraham Lincoln/        ← the folder's name is their primary name
    1.jpg  2.jpg  …
    person.json           ← optional: {"names": ["Honest Abe"], "describe": ["the 16th president"], "anchor": "a man"}
  Frederick Douglass/
    1.jpg  …
```

Every person needs photos: the trained stage anchors their erase loss on their own latents. Loose
images directly in the folder are refused — a guard has to know whose likeness it closes. More than
one person requires `--consent representative`; "I am this person" cannot attest for anyone else.

After the export a **quick check** runs by default: each name's pull on that person's held-out
photos with the guard off vs on, plus drift on everyday prompts. It is not a verdict — there is no
invented-name null and no attack — so the verdicts here are their own:

| Verdict | Meaning |
|---|---|
| `quick-checked` | every name's pull fell below the residual tolerance and drift stayed inside it |
| `quick-partial` | a name still pulls, or everyday prompts moved too much |
| `unverified` | `--no-check`: the file was written and nothing measured it |
| `guarded` / `partial` / … | `--full-verify`: the real verifier ran, per person |

A guard fitted this way says nothing about what the model could reach. `veil verify --guard <file>`
measures that, one person at a time.

**Klein options** (`--option key=value`)

| Option | Meaning |
|---|---|
| `text-encoder` | `repo` (default: the linked repo's bf16 Qwen3, as hosts run it) or `mflux` (the shared 4-bit export: no download, slightly different embeddings, flagged) |
| `export-dir` | The shared mflux export for the VAE encoder (default `VEIL_FLUX2_DIR` or `~/Documents/huggingface/models/mlx-community/flux2-klein-4b-4bit`; downloaded if missing) |
| `quantize` | 4 or 8: re-quantize a bf16 transformer on load |
| `long-side` | Photo working size (default 512) |
| `max-batch` | Rows per transformer call (default 2) |
| `steps` | Sampler steps that define the σ grid (default 4 for distilled Klein, 50 for the base) |

**The app.** Run `scripts/run-app.sh`; `--release` makes it much faster on real models, `--detach` returns to the shell.
1. Paste a model link.
2. Drop photos of the person. **Add Person** adds another to the same guard.
3. Add names, descriptions and a controls folder.
4. Confirm the consent attestation.
5. Press **Assess + Guard** (one person, measured end to end) or **Build Guard** (one person or a
   group, straight to the guard, then the quick check).

The results show:
- a reachability table;
- before/after pull maps on the held-out photos;
- the guard's fit;
- verification cards;
- the issues and limits.

Save the guard, the report and the figure from there. **Forget This Person** deletes what Veil stored.

## How it works

Nothing is generated. Veil measures how much a prompt helps the model denoise **the person's own photos**, in the model's latent space, on the noise levels its sampler actually visits. A step-by-step account with every choice justified is in [Docs/METHOD.md](Docs/METHOD.md).

**1. Find.**
- **Pull.** A prompt's pull is the conditional-likelihood gain over the same prompt with the name replaced by an anchor:

      x_σ = (1−σ)x₀ + σε,  u = ε − x₀,
      ℓ(c) = face-weighted mean (v̂(x_σ,σ,c) − u)²,
      Pull = E[ℓ(anchor) − ℓ(c)]

  It is paired on the same keyed noise.
- **Specificity.** Spec = the person's held-out pull minus the pull on the **nearest control identity** (the look-alike the route pulls most). This keeps shared attributes out of it.
- **Calibration.** 19 invented names give p = (1 + #null ≥ Spec)/(1 + N). A route **reaches** when p ≤ 0.05 and the person's pull is positive at 95%.
- **Capability class:** `name-bound`, `description-reachable`, `search-reachable` or `not-reachable-at-budget`.

**2. Block.**
- **Stage A** is a closed-form edit of the slot that reads the text encoder's output (UCE-style). It is solved with a per-channel ridge, because LLM text encoders have massive-activation channels. It is then factored by reduced-rank regression in the data's own metric.
- **Stage B** trains a LoRA on the text path, anchored on the person's own photos: under every route, the guarded model should predict what the deployment predicts under the anchor. It has explicit preservation on controls, and optional adversarial hardening. Image-path weights are never touched.

**3. Verify.** Verification runs on the exported file, read back, with held-out photos and independent noise draws.
- **Suppression:** every route that reached is back inside the null.
- **Robustness:** the number of soft-embedding attack steps it takes to cross the null, on the base vs guarded.
- **Preservation:** velocity drift on control identities and on generic prompts, and each named control keeps its own-name pull.
- **Verdict:** `guarded` · `partial` · `not-needed` · `inconclusive`.

## The guard file

- **Formats.**
  - `diffusers` (default): PEFT names, `transformer.<module>.lora_A.weight` / `.lora_B.weight`. Diffusers and most hosts load these.
  - `bfl`: `diffusion_model.<bfl module>.lora_A/B.weight`, for ComfyUI and BFL tooling. The text stream's q/k/v are fused exactly by concatenating ranks.
- **Scale.** The scale is folded in, so apply it at 1.0.
- **Header metadata:**
  - `veil.base_model` (repo@commit), `veil.family`, `veil.variant`
  - `veil.subject` (a keyed hash, not a name)
  - `veil.consent` (basis and time)
  - `veil.method`, `veil.run_id`, `veil.intended_use`

## Consent and custody

- **Consent is required.** The CLI refuses without `--consent`, and the app needs the attestation box. It is recorded as a basis and a time, with no personal data.
- **Photos stay put.** They never leave the machine and are never copied. Reports carry their SHA-256 hashes.
- **What Veil stores.** Only latents and the sealed list of discovered prompts, under `~/Library/Application Support/Veil/subjects/<id>`. `veil forget` deletes it.
- **Discovered prompts** are attack recipes. They are listed by hash unless you pass `--reveal-routes`.
- **The pull figure** shows the person's photos and is local only.

## Results so far

**Toy world** (`veil research toy`; the answer is known). Ada Quill is reachable by her name and by one description; Cleo Marsh shares her red hair.

| Method | Verdict | Details |
|---|---|---|
| closed form | GUARDED | Name and description suppressed; max drift 0.06%; soft-attack steps to cross the null: name 0 → 50, anchor 25 → 50 |
| closed form + trained | GUARDED | Same suppression; max drift 0.16% |

In both runs:
- the look-alike's own-name pull is kept (≥ 97%);
- the attribute prompt ("a red-haired person") still works.

**A group in one guard** (`veil protect`, same world). Ada Quill and Bram Oake protected together, with Cleo Marsh and Dov Reyes as controls:

| Check | Result |
|---|---|
| Both protected people | `guarded` — specificity 1.18 → 0.00 and 1.33 → −0.00, each back inside the null |
| The two controls | Cleo Marsh and Dov Reyes keep 100% of their own-name pull |
| Collateral and cost | Max drift 0.04%; the name attack costs 0 → 25 steps |

**FLUX.2 klein-base-4B**, checked on the real weights:

| Check | Result |
|---|---|
| Guard off vs deployment | Bitwise identical; with the guard on, the output changes |
| Text encoder | The input-embedding path matches the id path |
| diffusers ↔ BFL exports | Read back to the same deltas, 100% mapped |
| Closed-form edit of `context_embedder` | At rank 64 it leaves ≈ 25% of the name→anchor gap energy (half the gap in amplitude), drift under 1%. Two people sharing one edit leaves 28–43% each. Measured over 8 noise draws: a single draw ranges from 8% to 88% left, so single-draw numbers for this edit are meaningless. |
| Sampled positive control (Lincoln, the model's own samples) | `partial`. The guard cut the name's pull by 74% (0.0231 → 0.0060) and its specificity to zero, but the run also showed the scoring's limits: with 9 nulls at α = 0.1, p = 0.10 is the floor, and a guarded name sits *above* a null cloud that is entirely negative, so suppression cannot be declared however well it worked. Details and the σ breakdown in [Docs/VALIDATION.md](Docs/VALIDATION.md). |

## Package layout

```
Sources/VeilKit/
  Veil.swift            VeilRun: prepare · assess · fitGuard · export · verify · report · figure
  Protect.swift         ProtectRun: block without find — one person or a group, then the quick check
  Model/                executor protocols (PhotoEncoder, PromptEmbedder, VelocityModel, GuardSlots, AdapterNaming,
                        TokenSearchInterface), HookedLinear, adapter files, family detection, ExecutorRegistry
  Subject/              photos, Vision face segmentation → likeness weights, consent, controls, split, custody store
  Routes/               templates, null names, soft-prompt attack, discrete prompt search (PEZ), locator
  Assess/               draws, pull, reachability (nearest-control specificity, null calibration)
  Guard/                what to erase and preserve (GuardPlan/GuardFitter), closed-form edit + reduced-rank
                        factorization, trainer (erase pairs matched to their own person), Adam
  Verify/               suppression, robustness, preservation, verdict; the quick check for unassessed guards
  Export/ Report/       LoRA writer and read-back, veil.guard.v1 report, pull figure
  Source/ Fetch/ Formats/ Seeds/ Support/   links, range fetching, safetensors, keyed seeds, profiles
  Toy/                  the analytic identity world and its executor
Sources/VeilFlux2/      Klein executor: components, key map (import + diffusers/BFL export), token search, registration
Sources/veil/           CLI (composition root)
Sources/VeilApp/        main.swift → NSApplication, AppKit menu, SwiftUI views
```

## Limits

- **Routes, not the likeness.** A guard is only as complete as the routes searched, and it holds only at the stated attack budgets.
- **Deployment only.** The block binds only where the deployer applies it.
- **Pull is a proxy.** It is a decode-free conditional-likelihood proxy.
- **Uncalibrated thresholds.** α, the drift tolerance and the retained-pull threshold are uncalibrated until the validation protocol ([Docs/VALIDATION.md](Docs/VALIDATION.md)) has run on real people with known answers.
- **One executor.** FLUX.2 Klein is the only executor so far. `inspect` names other families and reports them as unsupported.

## License

GPL-3.0; see [LICENSE](LICENSE). Parts are adapted from Scorpion (GPL-3.0); file headers say which.
