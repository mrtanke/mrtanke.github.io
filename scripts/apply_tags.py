import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"

ALLOWED_TAGS = {
    "vision-model",
    "language-model",
    "vision-language-model",
    "vision-language-action",
    "world-model",
    "reinforcement-learning",
    "fine-tuning",
    "embodied-intelligence",
    "light-field",
    "rag",
    "hardware",
    "efficient-llms",
    "state-space-models",
    "super-resolution",
    "compilers",
    "reasoning",
    "optimization",
    "attention",
}

TAG_MAP = {
    "content/posts/2025-12-09-alta-compiler-based-analysis-of-transformers/index.md": [
        "compilers",
        "attention",
        "reasoning",
    ],
    "content/posts/2025-11-29-atlas-learning-to-optimally-memorize-the-context-at-test-time/index.md": [
        "language-model",
        "efficient-llms",
        "attention",
        "world-model",
    ],
    "content/posts/2025-12-04-linformer-self-attention-with-linear-complexity/index.md": [
        "attention",
        "efficient-llms",
    ],
    "content/posts/2025-12-16-synthesizer-rethinking-self-attention-for-transformer-models/index.md": [
        "attention",
        "efficient-llms",
    ],
    "content/posts/2025-11-07-reference-based-face-super-resolution-using-the-spatial-transformer/index.md": [
        "vision-model",
        "super-resolution",
    ],
    "content/posts/2025-11-25-roformer-enhanced-transformer-with-rotary-position-embedding/index.md": [
        "attention",
        "efficient-llms",
    ],
    "content/posts/2025-11-30-what-formal-languages-can-transformers-express-a-survey/index.md": [
        "language-model",
        "reasoning",
        "attention",
    ],
    "content/posts/2025-12-08-tracr-compiled-transformers-as-a-laboratory-for-interpretability/index.md": [
        "compilers",
        "attention",
        "reasoning",
    ],
    "content/posts/2025-11-06-latent-diffusion-models/index.md": [
        "vision-model",
    ],
    "content/posts/2025-11-09-mixture-of-recursions-learning-dynamic-recursive-depths-for-adaptive-token-level-computation/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-11-04-deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning/index.md": [
        "language-model",
        "reasoning",
        "fine-tuning",
        "reinforcement-learning",
    ],
    "content/posts/2025-11-01-a-tutorial-on-bayesian-optimization/index.md": [
        "optimization",
    ],
    "content/posts/2025-10-21-crossnet-an-end-to-end-reference-based-super-resolution-network-using-cross-scale-warping/index.md": [
        "vision-model",
        "super-resolution",
    ],
    "content/posts/2025-11-10-exploiting-spatial-and-angular-correlations-with-deep-efficient-transformers-for-light-field-image-super-resolution/index.md": [
        "light-field",
        "super-resolution",
        "vision-model",
    ],
    "content/posts/2025-11-07-lmr-a-large-scale-multi-reference-dataset-for-reference-based-super-resolution/index.md": [
        "vision-model",
        "super-resolution",
        "light-field",
    ],
    "content/posts/2025-11-14-a-survey-for-light-field-super-resolution/index.md": [
        "light-field",
        "super-resolution",
    ],
    "content/posts/2025-11-11-efficiently-modeling-long-sequences-with-structured-state-spaces/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-11-28-solving-olympiad-geometry-without-human-demonstrations/index.md": [
        "reasoning",
        "language-model",
    ],
    "content/posts/2025-11-03-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/index.md": [
        "vision-model",
        "attention",
    ],
    "content/posts/2025-12-10-random-search-for-hyper-parameter-optimization/index.md": [
        "optimization",
    ],
    "content/posts/2025-11-27-formal-mathematical-reasoning-a-new-frontier-in-ai/index.md": [
        "reasoning",
        "language-model",
    ],
    "content/posts/2025-12-10-bayesian-optimization-is-superior-to-random-search-for-machine-learning-hyperparameter-tuning/index.md": [
        "optimization",
    ],
    "content/posts/2025-11-17-mamba-linear-time-sequence-modeling-with-selective-state-spaces/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-10-29-crossnet-cross-scale-large-parallax-warping-for-reference-based-super-resolution/index.md": [
        "vision-model",
        "super-resolution",
    ],
    "content/posts/2025-10-01-attention-is-all-you-need/index.md": [
        "attention",
        "language-model",
    ],
    "content/posts/2025-12-28-diffusion-policy-visuomotor-policy-learning-via-action-diffusion/index.md": [
        "vision-language-action",
        "embodied-intelligence",
    ],
    "content/posts/2025-10-10-a-bridging-model-for-parallel-computation/index.md": [
        "hardware",
        "attention",
    ],
    "content/posts/2025-10-27-rwkv-reinventing-rnns-for-the-transformer-era/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-12-05-fnet-mixing-tokens-with-fourier-transforms/index.md": [
        "attention",
        "efficient-llms",
    ],
    "content/posts/2025-11-11-retentive-network-a-successor-to-transformer-for-large-language-models/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-11-18-hyena-hierarchy-towards-larger-convolutional-language-models/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-12-01-on-the-representational-capacity-of-neural-language-models-with-chain-of-thought-reasoning/index.md": [
        "reasoning",
        "language-model",
    ],
    "content/posts/2025-10-20-chain-of-thought-prompting-elicits-reasoning-in-large-language-models/index.md": [
        "reasoning",
    ],
    "content/posts/2025-12-03-rethinking-attention-with-performers/index.md": [
        "attention",
        "efficient-llms",
    ],
    "content/posts/2025-11-19-disentangling-light-fields-for-super-resolution-and-disparity-estimation/index.md": [
        "light-field",
        "super-resolution",
    ],
    "content/posts/2025-10-16-from-local-to-global-a-graphrag-approach-to-query-focused-summarization/index.md": [
        "rag",
        "reasoning",
    ],
    "content/posts/2025-11-24-mastering-the-game-of-go-without-human-knowledge/index.md": [
        "reinforcement-learning",
        "world-model",
    ],
    "content/posts/2025-10-20-learningbased-light-field-imaging/index.md": [
        "light-field",
        "vision-model",
    ],
    "content/posts/2025-10-28-xlstm-extended-long-short-term-memory/index.md": [
        "state-space-models",
        "efficient-llms",
    ],
    "content/posts/2025-11-26-titans-learning-to-memorize-at-test-time/index.md": [
        "language-model",
        "efficient-llms",
    ],
    "content/posts/2025-11-24-mastering-chess-and-shogi-by-self-play-with-a-general-reinforcement-learning-algorithm/index.md": [
        "reinforcement-learning",
    ],
    "content/posts/2025-12-06-its-all-connected-a-journey-through-test-time-memorization-attentional-bias-retention-and-online-optimization/index.md": [
        "language-model",
    ],
    "content/posts/2025-12-14-reformer-the-efficient-transformer/index.md": [
        "attention",
        "efficient-llms",
    ],
    "content/posts/2025-12-07-thinking-like-transformers/index.md": [
        "reasoning",
        "attention",
        "language-model",
    ],
    "content/posts/2025-12-12-openvla-an-open-source-vision-language-action-model/index.md": [
        "vision-language-action",
        "vision-language-model",
        "embodied-intelligence",
    ],
    "content/posts/2025-10-15-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/index.md": [
        "rag",
        "language-model",
    ],
    "content/posts/2025-12-15-learning-transformer-programs/index.md": [
        "compilers",
        "attention",
        "reasoning",
    ],
    "content/posts/2025-10-24-mastering-the-game-of-go-with-mcts-and-deep-neural-networks/index.md": [
        "reinforcement-learning",
        "reasoning",
    ],
    "content/projects/2026-01-02-reproducing-diffusion-policy/index.md": [
        "embodied-intelligence",
        "vision-language-action",
    ],
}


def update_file(path: Path, tags: list[str]) -> None:
    if not set(tags).issubset(ALLOWED_TAGS):
        missing = set(tags) - ALLOWED_TAGS
        raise ValueError(f"{path}: unknown tags {missing}")

    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ValueError(f"{path} does not have standard front matter")

    front_matter_raw, body = match.groups()
    data = yaml.safe_load(front_matter_raw) or {}
    data.pop("tag", None)
    data["tags"] = tags

    new_front = yaml.safe_dump(data, sort_keys=False).rstrip()
    path.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")


def main() -> None:
    for rel_path, tags in TAG_MAP.items():
        path = ROOT / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        update_file(path, tags)


if __name__ == "__main__":
    main()
