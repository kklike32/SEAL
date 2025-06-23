"""
Generate synthetic SQuAD-style items by prompting an MLX model for `k` "implication" completions per passage.
"""
from pathlib import Path
import argparse
import json
import random
import time
import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

# Reuse the same prompt templates from the original script
MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant tasked with analyzing the provided passage and producing a list of implications derived directly or indirectly from the content. <|im_end|>\n"
    "<|im_start|>user\n{title}\n{context}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

MAKE_SQUAD_DATA_TEMPLATES_BASE: dict[str, str] = {
    # list of implications
    "implications": (
        "Let's read the following passage and produce a list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # long list of implications
    "implications-long": (
        "Let's read the following passage and produce a long list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # very long list of implications
    "implications-very-long": (
        "Let's read the following passage and produce a very long list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # rewrite the passage
    "rewrite": (
        "Let's read the following passage and rewrite it in a few different ways, each one separated by a newline.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Rewritten passages:\n"
    ),

    # self-qa
    "self-qa": (
        "Let's read the following passage and rewrite it in a question-answer format.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Question 1: "
    ),
}

# ------------------------------------------------------------------------ #

def make_prompt(title: str, context: str, instruct_model: bool, prompt_key: str) -> str:
    MAKE_SQUAD_DATA_TEMPLATE = MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT if instruct_model else MAKE_SQUAD_DATA_TEMPLATES_BASE[prompt_key]
    return MAKE_SQUAD_DATA_TEMPLATE.format(
            title=title,
            context=context,
        )

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to the MLX model")
    p.add_argument("--instruct_model", action="store_true", help="Using instruction model")
    p.add_argument("--dataset_in", required=True, help="Path to the input dataset")
    p.add_argument("--dataset_out_dir", required=True, help="Directory to save the output dataset")
    p.add_argument("--n", type=int, default=-1, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index for processing")
    p.add_argument('--k', type=int, default=5, help='Completions per article')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling (top-p)')
    p.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate")
    p.add_argument("--prompt_key", default="implications", choices=list(MAKE_SQUAD_DATA_TEMPLATES_BASE.keys()), help="Which prompt to use")
    args = p.parse_args()

    # Load the MLX model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)
    print("Model loaded successfully.")

    # Load the data
    raw = json.load(open(args.dataset_in, encoding="utf-8"))
    random.seed(42)
    random.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start:]

    out_data = []
    t0 = time.time()
    for i, item in enumerate(subset):
        print(f"Processing item {i+1}/{len(subset)}: {item['title']}")
        prompt = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key)
        
        completions = []
        for _ in range(args.k):
            sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
            completion = generate(model, tokenizer, prompt, sampler=sampler, max_tokens=args.max_tokens)
            completions.append(completion)

        new_item = dict(item)
        new_item["completions"] = completions
        new_item["prompt"] = prompt
        out_data.append(new_item)

    print(f"Generated {len(out_data) * args.k} completions in {time.time()-t0:.1f}s")

    # Save the results
    out_path = Path(args.dataset_out_dir) / f"{Path(args.dataset_in).stem}_mlx_generated.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")

    # Save metadata
    meta = {
        "model": args.model_path,
        "dataset_in": args.dataset_in,
        "dataset_out": str(out_path),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": len(subset),
        "k": args.k,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {meta_path}")

if __name__ == "__main__":
    main()
