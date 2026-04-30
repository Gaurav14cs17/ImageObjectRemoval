# How the project is organized

This is a plain-English walkthrough of every folder. Read it once and you'll know where everything lives.

Here's a quick preview of what the project does — input on the left, output on the right:

| Input | Output |
|:---:|:---:|
| ![input](../assets/examples/input_1.png) | ![output](../assets/examples/output_1.png) |

---

## The root — setup files

When you first open the repo, you'll see a few files at the top level. These are all about setting up your environment, nothing fancy:

- **`README.md`** — The main page. Explains what the project does, how to set it up, how to use it.
- **`requirements.txt`** — List of Python packages you need. You run `pip install -r requirements.txt` once and you're good.
- **`pyproject.toml`** — This lets you do `pip install -e .` so Python treats the `omnieraser/` folder as a real installable package. You only need to run this once.
- **`.github/`** — GitHub Actions config. Right now it just deploys the website to GitHub Pages. You probably won't touch this.

---

## `omnieraser/` — the brain of the project

This is where the actual model code lives. You never run these files directly — you import from them.

There are only three files:

- **`__init__.py`** — The package entry point. It just re-exports the two main things so you can write `from omnieraser import FluxControlRemovalPipeline` instead of the full path.

- **`pipeline_flux_control_removal.py`** — This is the big one. It's the diffusion pipeline that actually does the object removal. It takes your image and mask, encodes them into latent space, runs the denoising loop, and decodes the result back into a normal image. If you want to understand how the model works, this is the file to read.

- **`utils.py`** — A small helper file. Right now it just has `PairedRandomCrop`, which is used during training to randomly crop the image and its mask together (so they stay aligned). Nothing complicated.

**When would you edit these files?** When you want to change how the model behaves — add new post-processing, change how the mask is handled, modify the denoising logic, etc.

---

## `scripts/` — things you actually run

This is where all the runnable stuff lives. You open your terminal, type a command, and one of these scripts does something.

- **`test_control_lora_flux.py`** — The simplest one. Loads the model, runs it on one example image, saves the result. This is your "does everything work?" check. About 70 lines, easy to follow.

- **`gradio_control_lora_flux.py`** — Launches a web UI on your machine. You upload a photo, draw over the thing you want removed, click a button, and get the cleaned image. Runs at `http://localhost:7999`.

- **`train_control_lora_flux.py`** — The training script. If you have your own dataset, this is how you fine-tune the LoRA adapter. It handles everything: loading data, setting up the optimizer, running the training loop, saving checkpoints.

- **`train_control_lora_flux.sh`** — A shell script that calls `accelerate launch` with all the right arguments for training. Think of it as a shortcut so you don't have to type a giant command every time.

**Important:** Always run these from the repo root folder, not from inside `scripts/`. Like this:

```bash
python scripts/test_control_lora_flux.py      # correct
python test_control_lora_flux.py              # won't work (wrong directory)
```

---

## `configs/` — training settings

Just one file here:

- **`accelerate.yaml`** — Tells the training script how many GPUs to use, what precision to run in (bf16), and other distributed training settings. The shell script (`train_control_lora_flux.sh`) points to this file automatically.

**When would you edit this?** When you switch machines, change your GPU count, or want to try a different precision setting.

---

## `example/` — sample images to play with

Two sub-folders:

- **`example/image/`** — A bunch of input photos (PNG files).
- **`example/mask/`** — The matching masks. Each mask has the same filename as its image. White areas = the object to remove. Black areas = keep as-is.

The test script and the Gradio example gallery both pull from here. If you want to add your own test images, just drop them in with matching filenames.

---

## `ControlNet_version/` — the other approach

This is a completely separate variant of the project that uses ControlNet instead of plain LoRA. It has its own pipeline files, its own training script, its own Gradio demo, and even its own `requirements.txt`.

Think of it as a sibling project. The idea is the same (remove objects from images), but the architecture is a bit different and it gives better background consistency in some cases.

You can totally ignore this folder if you're just using the main (base) version. If you want to try it, go into the folder and treat it like its own mini-project.

---

## `web/` — the project website

- **`index.html`** — A static landing page for the project.
- **`static/`** — CSS, JavaScript, and images for that page.

This gets deployed to GitHub Pages. It has nothing to do with the Python code — it's just a nice web page to show off results.

---

## How the pieces connect (flow diagram)

This shows which files depend on what, so you can see the big picture:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                      Hugging Face Hub                            │
  │                (FLUX.1-dev + LoRA weights)                       │
  └───────┬─────────────────────┬────────────────────┬───────────────┘
          │                     │                    │
          v                     v                    v
  ┌───────────────┐   ┌─────────────────┐   ┌────────────────────┐
  │ test_control_ │   │ gradio_control_ │   │ train_control_     │
  │ lora_flux.py  │   │ lora_flux.py    │   │ lora_flux.py       │
  │ (quick test)  │   │ (web UI)        │   │ (training)         │
  └──┬──────┬─────┘   └──┬──────┬───────┘   └──┬──────┬─────┬───┘
     │      │             │      │              │      │     │
     │      └─────────────┴──────┘              │      │     │
     │            imports from                  │      │     │
     v                                          v      │     │
  ┌────────────────────────────┐   ┌──────────────┐    │     │
  │ omnieraser/                │   │ omnieraser/   │    │     │
  │ pipeline_flux_control_     │   │ utils.py      │    │     │
  │ removal.py                 │   └───────────────┘    │     │
  └────────────────────────────┘                        │     │
                                                        │     │
  ┌──────────────────┐      ┌──────────────────────┐    │     │
  │ example/         │      │ configs/             │    │     │
  │ image/ + mask/   │      │ accelerate.yaml      │────┘     │
  │ (sample data)    │      │ (GPU & precision)    │          │
  └──────────────────┘      └──────────────────────┘          │
                                                              │
                            ┌──────────────────────┐          │
                            │ train_control_        │──────────┘
                            │ lora_flux.sh          │
                            │ (launch command)      │
                            └───────────────────────┘
```

Read it like this: lines going down mean "uses" or "depends on." For example, `test_control_lora_flux.py` pulls weights from Hugging Face, loads sample images from `example/`, and imports the pipeline from `omnieraser/`.

---

## What to read and in what order

If you're new to this project and want to understand everything, here's the path I'd recommend:

1. **This file** — you're here, so you already know where everything is.
2. **`scripts/test_control_lora_flux.py`** — Read the whole thing. It's short and shows you the complete flow from start to finish.
3. **`omnieraser/pipeline_flux_control_removal.py`** — The core pipeline. Read it after the test script so you have context for what each method does.
4. **`omnieraser/utils.py`** — Quick read. Just one class.
5. **`scripts/train_control_lora_flux.py`** — The training loop. It's long, but most of it is standard HuggingFace boilerplate. Focus on the data loading and the loss computation.
6. **`scripts/gradio_control_lora_flux.py`** — Good to see how the pipeline gets wrapped in a UI.

---

## Technical docs

For the deeper stuff, there are dedicated docs in this same folder:

| Doc | What it covers |
|-----|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Every component (VAE, transformer, LoRA, text encoders, scheduler), tensor shapes at each stage, the channel expansion trick, LoRA target modules |
| [INFERENCE.md](INFERENCE.md) | Line-by-line walkthrough of what happens when you run inference — from loading the model to the final decoded image |
| [TRAINING.md](TRAINING.md) | Dataset format, data augmentation, loss function, all hyperparameters, how to launch and resume training, memory requirements |

---

## Cheat sheet

| I want to... | Where to go |
|---|---|
| See if the model works | `python scripts/test_control_lora_flux.py` |
| Remove objects with a UI | `python scripts/gradio_control_lora_flux.py` |
| Train on my own data | Edit `scripts/train_control_lora_flux.sh`, then `bash scripts/train_control_lora_flux.sh` |
| Understand the pipeline | Read `omnieraser/pipeline_flux_control_removal.py` |
| Understand the full architecture | Read [docs/ARCHITECTURE.md](ARCHITECTURE.md) |
| Understand inference step by step | Read [docs/INFERENCE.md](INFERENCE.md) |
| Understand training in detail | Read [docs/TRAINING.md](TRAINING.md) |
| Change GPU/precision settings | Edit `configs/accelerate.yaml` |
| Add my own test images | Drop them in `example/image/` and `example/mask/` |
| Try the ControlNet version | Go into `ControlNet_version/` |
