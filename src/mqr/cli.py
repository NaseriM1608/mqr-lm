"""Command-line entry points for the first controlled MQR experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import benchmark_model
from .config import ExperimentConfig
from .data import ByteTokenizer, TokenStream, load_tokenizer
from .model import LanguageModel
from .provenance import prepared_identity, raw_byte_identity, validate_tokenizer
from .training import evaluate, load_model, run_training, set_seed


def _write_result(result: dict, output: str | None) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _streams(args, config):
    length, vocab = config.model.context_length, config.model.vocab_size
    if args.data_dir:
        if args.train_text or args.validation_text:
            raise ValueError("Choose --data-dir or text files, not both.")
        root = Path(args.data_dir)
        identity = prepared_identity(root)
        tokenizer = load_tokenizer(root / "tokenizer.json")
        if tokenizer.vocab_size != vocab:
            raise ValueError(f"Prepared tokenizer has {tokenizer.vocab_size} entries, model has {vocab}.")
        return (TokenStream.from_file(root / "train.npy", length, vocab),
                TokenStream.from_file(root / "validation.npy", length, vocab), identity)
    if not args.train_text or not args.validation_text:
        raise ValueError("Provide --data-dir, or both --train-text and --validation-text.")
    if vocab != ByteTokenizer().vocab_size:
        raise ValueError("Raw-text smoke mode requires vocab_size=259; prepare BPE data for research configs.")
    return (TokenStream.from_text(args.train_text, length), TokenStream.from_text(args.validation_text, length), raw_byte_identity())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mqr", description="MQR-v1 implementation and controlled baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train with exact full-vocabulary CE")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--data-dir")
    train.add_argument("--train-text")
    train.add_argument("--validation-text")
    train.add_argument("--resume")
    train.add_argument("--stop-after", type=int, help="Pause after this absolute optimizer step without changing the full schedule")
    train.add_argument("--output")

    ev = sub.add_parser("evaluate", help="Evaluate exact full-vocabulary NLL")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--tokens", help="Held-out .npy token stream")
    ev.add_argument("--data-dir", help="Verified prepared bundle, required for BPE evaluation")
    ev.add_argument("--split", choices=("validation", "test"), default="validation")
    ev.add_argument("--text", help="Held-out byte-tokenized smoke text")
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--batches", type=int, default=16)
    ev.add_argument("--batch-size", type=int, default=2)
    ev.add_argument("--output")

    gen = sub.add_parser("generate", help="Generate with exact or retrieved head scores and a KV cache")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--tokenizer", help="Prepared tokenizer.json; defaults to smoke byte tokenizer")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--max-new-tokens", type=int, default=16)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-k", type=int, help="Sampling limit within the available token set")
    gen.add_argument("--method", choices=("exact", "flat", "faiss", "hnsw"), default="exact")
    gen.add_argument("--k", type=int, default=32, help="Retrieved candidates per query; used by retrieval methods")
    gen.add_argument("--seed", type=int, default=0)

    bench = sub.add_parser("benchmark", help="Compare exact and retrieved greedy decode timing")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--tokenizer")
    bench.add_argument("--prompt", default="The next token is")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--new-tokens", type=int, default=8)
    bench.add_argument("--method", choices=("flat", "faiss", "hnsw"), default="flat")
    bench.add_argument("--k", type=int, default=32)
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--threads", type=int, default=2)
    bench.add_argument("--output")

    inspect = sub.add_parser("inspect", help="Validate a runnable config and count model parameters")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = ExperimentConfig.from_json(args.config)
            train_stream, validation_stream, identity = _streams(args, config)
            result = run_training(config, train_stream, validation_stream, args.run_dir,
                                  resume=args.resume, stop_after=args.stop_after, data_identity=identity)
            _write_result(result, args.output)
        elif args.command == "inspect":
            config = ExperimentConfig.from_json(args.config)
            set_seed(config.training.seed)
            model = LanguageModel(config)
            _write_result({"config": config.to_dict(), "parameters": sum(p.numel() for p in model.parameters()),
                           "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
                           "output_head_parameters": sum(p.numel() for p in model.head.parameters()),
                           "batch_tokens_per_optimizer_step": config.training.batch_size *
                           config.model.context_length * config.training.gradient_accumulation,
                           "planned_predicted_tokens": config.training.batch_size * config.model.context_length *
                           config.training.gradient_accumulation * config.training.max_steps}, args.output)
        else:
            model, config, payload = load_model(args.checkpoint, args.device)
            torch.set_num_threads(config.training.num_threads)
            if args.command == "evaluate":
                if sum(map(bool, (args.tokens, args.text, args.data_dir))) != 1:
                    raise ValueError("Provide exactly one of --data-dir, --tokens or --text.")
                if args.text and config.model.vocab_size != 259:
                    raise ValueError("--text evaluation is only for the byte-tokenized smoke model.")
                identity = payload.get("data_identity", {})
                if args.data_dir:
                    current_identity = prepared_identity(args.data_dir)
                    if current_identity["tokenizer_sha256"] != identity.get("tokenizer_sha256"):
                        raise ValueError("Evaluation tokenizer does not match the checkpoint.")
                    stream = TokenStream.from_file(Path(args.data_dir) / f"{args.split}.npy", config.model.context_length, config.model.vocab_size)
                else:
                    validate_tokenizer(identity, None)
                    stream = (TokenStream.from_file(args.tokens, config.model.context_length, config.model.vocab_size)
                              if args.tokens else TokenStream.from_text(args.text, config.model.context_length))
                result = evaluate(model, stream, batches=args.batches, batch_size=args.batch_size,
                                  chunk_positions=config.training.loss_chunk_positions)
                _write_result(result, args.output)
            else:
                validate_tokenizer(payload.get("data_identity", {}), args.tokenizer)
                tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else ByteTokenizer()
                if tokenizer.vocab_size != config.model.vocab_size:
                    raise ValueError("Pass the tokenizer.json matching the checkpoint vocabulary.")
                ids = tokenizer.encode(args.prompt, add_eos=False)
                if not ids:
                    ids = [tokenizer.bos_id]
                prompt = torch.tensor([ids], device=args.device, dtype=torch.long)
                if args.command == "generate":
                    set_seed(args.seed)
                    output = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k,
                                            tokenizer.eos_id, method=args.method, k=args.k)
                    print(tokenizer.decode(output[0].tolist()))
                else:
                    if args.batch_size <= 0 or args.threads <= 0:
                        raise ValueError("Batch size and thread count must be positive.")
                    torch.set_num_threads(args.threads)
                    prompt = prompt.expand(args.batch_size, -1).clone()
                    result = benchmark_model(model, prompt, new_tokens=args.new_tokens, k=args.k,
                                             method=args.method, repeats=args.repeats,
                                             always_include=(tokenizer.eos_id,))
                    _write_result(result, args.output)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    return 0
