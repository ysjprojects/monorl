# Data

Datasets are described by `data:` in the `RunConfig` and resolved by `monorl.data.open_source`
into a stream of `Sample(id, messages, reference, metadata)`. The web UI's dataset picker and
`monorl data inspect` use the same code (`monorl.data.browse.preview_uri`), so what you preview
is what training sees.

## URI schemes

| `data.uri` | Source | Notes |
|---|---|---|
| `hf://org/name` | Hugging Face Hub dataset | `data.split` (default `train`), `data.hf_config_name` for subsets (GSM8K: `main`). `hf://org/name@revision` pins a revision; `hf://org/name/path/in/repo.jsonl` narrows to specific files. `hf://datasets/org/name` is tolerated. Gated/private datasets use your `HF_TOKEN` / `huggingface-cli login`. |
| `s3://bucket/key.jsonl`, `s3://bucket/prefix/` | S3 or S3-compatible object store via `s3fs` | Credentials from the environment only (below). A prefix reads every supported file under it. |
| `gs://`, `az://`, … | any other fsspec filesystem | works when the driver (`gcsfs`, `adlfs`, …) is installed; credentials per that driver. |
| `/path/to/file.jsonl`, `./dir/`, `data/*.parquet`, `file://…` | local filesystem | file, directory or glob. |

## S3 credentials and endpoints

Nothing secret ever goes in a config file or the UI; `data.extra` rejects `key`/`secret`/`token`/
`storage_options`. Set the standard AWS variables in the environment that runs `monorl serve` or
`monorl train`:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | static credentials (omit both to use the botocore default chain: IAM role, `~/.aws/credentials`, SSO) |
| `AWS_SESSION_TOKEN` | temporary credentials (STS / SSO) |
| `AWS_DEFAULT_REGION` or `AWS_REGION` | bucket region |
| `AWS_ENDPOINT_URL` | custom endpoint for **MinIO** (`http://localhost:9000`), **Cloudflare R2** (`https://<account>.r2.cloudflarestorage.com`), Ceph RGW, LocalStack… |
| `AWS_PROFILE` | pick a profile from `~/.aws/config` (botocore reads it) |

`GET /api/connectors` (and the UI's S3 panel) reports `configured`, `region`, `endpoint_url` and
which variable names are present — never values. Quick check from the shell:

```
AWS_ENDPOINT_URL=http://localhost:9000 AWS_ACCESS_KEY_ID=minio AWS_SECRET_ACCESS_KEY=minio123 \
  monorl data inspect s3://my-bucket/data/train.jsonl --limit 3
```

See [`configs/recipes/s3_dataset_example.yaml`](../configs/recipes/s3_dataset_example.yaml).

## File formats

`.jsonl`/`.ndjson`, `.json` (array of objects), `.csv`/`.tsv`, `.parquet`, `.arrow`
(`datasets.save_to_disk` directories work), optionally compressed (`.gz`, `.bz2`, `.xz`, `.zst`).
Directories mixing formats read only the highest-priority one (parquet > arrow > jsonl > csv >
json). Readers stream, so previews and `data.streaming: true` never load whole files.

## Layouts and field mapping

`data.format` selects how rows become chats; `auto` (default) sniffs the columns of the first
rows (`sniff_format`):

| format | columns | result |
|---|---|---|
| `chat` | a messages column (`messages`, `conversations`, `conversation`; ShareGPT `from`/`value` is normalised) | `Sample.messages` = the list; the last assistant turn, if any, becomes the reference when no answer column exists |
| `prompt_answer` | prompt column (`prompt`, `question`, `instruction`, `input`, `problem`, `query`) + answer column (`answer`, `solution`, `response`, `output`, `target`, `label`, …) | `messages = [user: prompt]`, `reference = answer` |
| `text` | a lone prompt/text column | `messages = [user: text]`, `reference = None` |

Override the guess with `messages_field`, `prompt_field`, `answer_field`. Processing per row:

1. `prompt_template` — Python format string over the row's columns plus `{prompt}`:
   `"{prompt}\nThink step by step and end with '#### <answer>'."`
2. `system_prompt` — prepended as a `system` message.
3. Reference extraction — GSM8K-style `... #### 42` answers are reduced to `42`
   (the raw text stays under `metadata["answer_raw"]`). Rewards may extract further
   (`exact_match` `extract: last_number|boxed|after_hash|none`, `math_verify` parses LaTeX).
4. `max_prompt_tokens` — samples whose chat-templated prompt is longer are dropped (needs the
   tokenizer; the orchestrator passes the policy tokenizer).
5. `shuffle` (deterministic, `seed`) then `limit`.

`eval_uri`/`eval_split` describe an optional evaluation set with the same mapping.

## Inspecting

```
monorl data inspect hf://openai/gsm8k --config-name main --limit 3
monorl data inspect ./data/train.parquet --json
```

prints dtype per column, the inferred layout, the first rows and the `Sample`s monorl would
build. The UI's dataset page shows the same preview plus a column mapper that writes
`prompt_field`/`answer_field`/`messages_field` back into the config.
