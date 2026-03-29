# Image Quality Assessment Experiment

Evaluates two automated image quality assessment algorithms against generated images sampled from the database.

## Algorithms

### CLIP-IQA+

**What it measures:** Technical image quality — sharpness, noise levels, distortion, compression artifacts.

**Score range:** 0.0 to 1.0 (higher = better technical quality)

**Does NOT consider:** Prompt adherence, aesthetic preference, subject matter.

**Interpretation:**
| Score | Quality | Action |
|-------|---------|--------|
| < 0.3 | Very low — likely corrupted, heavily distorted, or failed generation | Recommend deletion |
| 0.3 – 0.5 | Poor — noticeable artifacts, blur, or noise | Review for deletion |
| 0.5 – 0.7 | Acceptable — minor quality issues | Keep unless space-constrained |
| > 0.7 | Good — technically sound image | Keep |

**Limitation:** Pure technical quality metric. A sharp, well-exposed image of the wrong subject still scores high.

### ImageReward

**What it measures:** Human preference alignment — how well the generated image matches its text prompt and general aesthetic quality as judged by human preference data.

**Score range:** -2.0 to +2.0 (raw), normalized to 0.0 – 1.0 (higher = better)

**Requires:** The text prompt that generated the image (read from .txt sidecar files).

**Interpretation:**
| Raw Score | Normalized | Quality | Action |
|-----------|-----------|---------|--------|
| < -1.0 | < 0.25 | Poor prompt alignment or aesthetics | Recommend deletion |
| -1.0 – 0.0 | 0.25 – 0.50 | Below average | Review for deletion |
| 0.0 – 0.5 | 0.50 – 0.625 | Average | Keep |
| > 0.5 | > 0.625 | Good alignment with prompt | Keep |

**Limitation:** Depends on prompt quality. A bad prompt scored against a faithful rendering still scores low.

## Combined Usage for Cleanup

- Use **CLIP-IQA+** as a fast filter for technically broken images (corrupted, blurry, heavy artifacts).
- Use **ImageReward** as a deeper quality signal for prompt-faithful, aesthetically pleasing images.
- **Recommend deletion** when both scores are low (logical AND reduces false positives):
  - CLIP-IQA+ normalized < 0.4 AND ImageReward normalized < 0.35
- For **agent auto-assessment**: use ImageReward as the primary signal (captures prompt adherence), with CLIP-IQA+ as a sanity check for technical failures.

## Usage

```bash
# Score 10 random images with both algorithms
./run/run_image_quality.sh

# Score images matching keywords
./run/run_image_quality.sh --query "woman,flowers" --k 20

# Score images from a specific folder
./run/run_image_quality.sh --folder "output/character__alice" --k 15

# Single algorithm only
./run/run_image_quality.sh --algorithm clip-iqa+ --k 5

# CPU mode (slower but no GPU required)
./run/run_image_quality.sh --device cpu --k 3
```

## Output

Results are written to `src/experiments/image_quality/output/` as timestamped text files containing:
1. Run metadata (query, folder, sample size)
2. Performance summary (overall time, per-algorithm model load time, scoring speed)
3. Per-image scores with timing
4. Ranking tables sorted by each algorithm's score

## Data Preparation

The experiment automatically samples images from the database and copies them to `data/images/` along with `.txt` sidecar files containing the generation prompt. This ensures:
- Both algorithms score the identical image set
- Prompts are available for ImageReward
- Users can inspect the sampled images and their prompts
