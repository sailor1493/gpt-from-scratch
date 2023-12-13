# Perplexity (PPL)
Perplexity (PPL) evaluation using the pretrained models.

## Dataset (NIKL)
We used 5 types of datasets from `National Institute of Korean Language (NIKL)` to measure perplexity.

| Source       | Dataset               |
|--------------|-----------------------|
| NIKL         | DIALOGUE_2020_v1.3    |
| NIKL         | KParlty_2021_v1.1     |
| NIKL         | NIKL_SPOKEN_v1.2      |
| NIKL         | WRITTEN_v1.2          |
| NIKL         | MESSENGER_v2.0        |

## Usage
Run the script file.
```bash
bash perplexity.sh
```

## Reference
- https://corpus.korean.go.kr/
- https://huggingface.co/docs/transformers/perplexity