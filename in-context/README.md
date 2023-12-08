# In-context Learning (KoBEST)
In-context few-shot learning performance evaluation on KoBEST datasets.

## Setup

```bash
# Requires python >= 3.9
git clone -b polyglot https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e . # Modify PyTorch ver. matching your CUDA ver. in 'setup.py' (or Reinstall PyTorch)
pip install evaluate importlib-resources
```
```bash
# (Important) Disable this warning!
export TOKENIZERS_PARALLELISM=false
```
 
## KoBEST Dataset
There are some minor differences with the `validation` set size from the paper.

| Tasks            | Train set | Dev set    | Test set | Metrics |
|------------------|:---------:|:----------:|:--------:|:-------:|
| kobest_boolq     | 3,667     | 700        | 1,404    | F1      |
| kobest_copa      | 3,076     | <U>500</U> | 1,000    | F1      |
| kobest_hellaswag | 2,029     | 500        | 500      | F1      |
| kobest_sentineg  | 3,649     | 400        | 397      | F1      |
| kobest_wic       | 3,318     | <U>610</U> | 1,260    | F1      |


## Evaluation

(Important) When evaluating `skt/kogpt2-base-v2` model, we need to add special tokens and remove assert function for tokenizer in the `lm-evaluation-harness/lm_eval/models/gpt2.py` file.

```bash
python main.py \
   --model gpt2 \
   --model_args pretrained=/data/s1/chanwoo/nlp_project/logs/bulk_books \
   --tasks kobest_copa,kobest_hellaswag,kobest_boolq,kobest_sentineg,kobest_wic \
   --num_fewshot 0 \
   --batch_size 16 \
   --device cuda:0 \
   --output_path /home/n0/yujin/repo/gpt-from-scratch/in-context/bulk_books
```

## Result

### BoolQ

Boolean Question Answering

```
{'paragraph': '두아 리파(Dua Lipa, 1995년 8월 22일 ~ )는 잉글랜드의 싱어송라이터, 모델이다. BBC 사운드 오브 2016 명단에 노미닛되었다. 싱글 "Be the One"가 영국 싱글 차트 9위까지 오르는 등 성과를 보여주었다.',
 'question': '두아 리파는 영국인인가?',
 'label': 1}
 ```

| **boolq(F1)** | 0-shot | 5-shot | 10-shot |
|:-------------:|:------:|:------:|:-------:|
|    baseline   | 0.3343 | 0.4951 |  0.4886 |
|   bulk_books  | 0.3343 | 0.3962 |  0.3958 |
|   newspaper   | 0.3343 | 0.4566 |  0.4308 |
|  specialized  | 0.3510 | 0.5026 |  0.4839 |
|     mixup     | 0.3343 | 0.5028 |  0.4914 |

### COPA

Choice of Plausible Alternatives

```
{'premise': '물을 오래 끓였다.',
 'question': '결과',
 'alternative_1': '물의 양이 늘어났다.',
 'alternative_2': '물의 양이 줄어들었다.',
 'label': 1}
```

| **copa(F1)** | 0-shot | 5-shot | 10-shot |
|:------------:|:------:|:------:|:-------:|
|   baseline   | 0.5855 | 0.5784 |  0.5636 |
|  bulk_books  | 0.5092 | 0.4960 |  0.5047 |
|   newspaper  | 0.5197 | 0.5165 |  0.5151 |
|  specialized | 0.4786 | 0.4813 |  0.4744 |
|     mixup    | 0.5113 | 0.5020 |  0.5003 |

### HellaSwag

HellaSwag

```
{'context': '모자를 쓴 투수가 타자에게 온 힘을 다해 공을 던진다. 공이 타자에게 빠른 속도로 다가온다. 타자가 공을 배트로 친다. 배트에서 깡 소리가 난다. 공이 하늘 위로 날아간다.',
 'ending_1': '외야수가 떨어지는 공을 글러브로 잡는다.',
 'ending_2': '외야수가 공이 떨어질 위치에 자리를 잡는다.',
 'ending_3': '심판이 아웃을 외친다.',
 'ending_4': '외야수가 공을 따라 뛰기 시작한다.',
 'label': 3}
```

| **hellaswag(F1)** | 0-shot | 5-shot | 10-shot |
|:-----------------:|:------:|:------:|:-------:|
|      baseline     | 0.3164 | 0.3168 |  0.3140 |
|     bulk_books    | 0.2787 | 0.2720 |  0.2731 |
|     newspaper     | 0.2588 | 0.2644 |  0.2490 |
|    specialized    | 0.2670 | 0.2673 |  0.2818 |
|       mixup       | 0.2911 | 0.2783 |  0.2757 |


### SentiNeg

Sentiment Negation Recognition

```
{'sentence': '택배사 정말 마음에 듬',
 'label': 1}
```

| **sentineg(F1)** | 0-shot | 5-shot | 10-shot |
|:----------------:|:------:|:------:|:-------:|
|     baseline     | 0.6158 | 0.4007 |  0.3771 |
|    bulk_books    | 0.4796 | 0.5226 |  0.4918 |
|     newspaper    | 0.5065 | 0.5365 |  0.5185 |
|    specialized   | 0.4032 | 0.4062 |  0.3415 |
|       mixup      | 0.3775 | 0.4775 |  0.3912 |

### WIC

Words-in-Context

```
{'word': '양분',
 'context_1': '토양에 [양분]이 풍부하여 나무가 잘 자란다.	',
 'context_2': '태아는 모체로부터 [양분]과 산소를 공급받게 된다.',
 'label': 1}
```

| **wic(F1)** | 0-shot | 5-shot | 10-shot |
|:-----------:|:------:|:------:|:-------:|
|   baseline  | 0.3328 | 0.4697 |  0.4715 |
|  bulk_books | 0.3280 | 0.3652 |  0.3657 |
|  newspaper  | 0.3280 | 0.4672 |  0.4564 |
| specialized | 0.3280 | 0.4832 |  0.5089 |
|    mixup    | 0.3280 | 0.4712 |  0.4756 |


## Reference
- https://huggingface.co/EleutherAI/polyglot-ko-1.3b#evaluation-results
- https://github.com/EleutherAI/lm-evaluation-harness/tree/polyglot
- https://huggingface.co/datasets/skt/kobest_v1 
- https://arxiv.org/pdf/2204.13509
