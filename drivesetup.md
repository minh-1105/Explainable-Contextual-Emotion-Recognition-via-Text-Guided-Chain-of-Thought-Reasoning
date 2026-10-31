# Google Drive Desktop Setup

The Git repository lives on the local machine. Colab notebooks read and write the Drive project:

```text
G:\My Drive\Colab Notebooks\Contextual Emotion Recognition
```

which Colab sees as `/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition`. Its layout is described in `workflow.md`.

## Directory mapping

Windows directory junctions in the local repository point at the same Drive folders the notebooks use:

| Local path | Google Drive target | Contents |
|---|---|---|
| `notebooks` | `notebooks` | Colab notebooks (tracked by Git) |
| `outputs` | `outputs` | `runs/` training runs and checkpoints, `vlm_pilot/` pilot runs |
| `models` | `models` | Exported models |

`data/` stays local; the EMOTIC annotations and image zips used by Colab are separate copies under the Drive `data/` folder. `.git` and `src/` stay local and must not be placed in Drive.

A junction is not a copy: saving or deleting a file through either path changes the same Drive-managed file.

## Git

`.gitignore` excludes `data/`, `outputs/`, `models/`, weights (`*.pt`, `*.pth`, `*.t7`) and archives (`*.zip`, `*.tar`, `*.tar.bz2`). Notebooks are committed normally.

## Verification

```powershell
$repoRoot = "C:\Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning"

Get-Item "$repoRoot\notebooks", "$repoRoot\outputs", "$repoRoot\models" |
Select-Object FullName, LinkType, Target
```

All three must report `LinkType` `Junction` with a target under `G:\My Drive`.

## Operating notes

- Google Drive for desktop must be running for streamed files to be accessible; mark the folder **Available offline** to use checkpoints without a network.
- Wait for Drive to finish syncing before shutting down or disconnecting a Colab runtime.
- Do not edit the same notebook in Colab and a local editor at the same time.
