# RecurrentMemForDocMT
## Data Processing 
Data processing follows the scripts from [G-transformer](https://github.com/baoguangsheng/g-transformer)

## File System
```bash
├── data
│   ├── ckpt 
│   │   ├──ckpt-sentence-europarl7
│   │   ├──ckpt-sentence-iwslt17
│   │   └──ckpt-sentence-nc2016
│   ├── europarl7.tokenized.en-de
│   │   ├──train/test/valid.de
│   │   └──train/test/valid.en
│   ├── iwslt17.tokenized.en-de
│   │   ├──train/test/valid.de
│   │   └──train/test/valid.en
│   ├── nc2016.tokenized.en-de
│   │   ├──train/test/valid.de
│   │   └──train/test/valid.en
│   ├── europarl7-vocab.txt
│   ├── iwslt17-vocab.txt
│   └── nc2016-vocab.txt
├── README.md
├── other files
```
We add the 4 special tokens at the beginning of each vocabulary file: &lt;pad&gt;, &lt;s&gt;, &lt;/s&gt;, &lt;unk&gt;    
Checkpoints refers to the well-trained sentence-level models.
## Package requirement
Before running the scripts, please install packages:
```bash
pip install transformers==4.10
```

## Train Sentence Baseline
```bash
python3 main.py --training-mode=sentence \
                --dataset=iwslt17 \
                --data-path=./data \
                --output-path=./output_filename \
                --mem-index= 6 \ # 6 for disable contextual memory
                --mem-length=16 \ # won't affect if mem-index is 6
                --mem-side=both \ # won't affect if mem-index is 6
                --mem-pos=sinusoidal \ # won't affect if mem-index is 6
                --batch-size=50 \ # won't affect if mem-index is 6, data is batched by total number of tokens in sentence-level training
                --dropout=0.3 \
                --dropout-mem=0.3 \ # won't affect if mem-index is 6
                --warmup-steps=4000 \ 
                --min-steps=20000  \
                --log-steps=1000 \
                --eval-steps=5000 \
                --learning-rate-sentence=0.0005 \
                --learning-rate-finetune=0.0003  \ # won't affect if mem-index is 6
                --min-optimize-step=1000 \  # won't affect if mem-index is 6
                --max-optimize-step=1000   # won't affect if mem-index is 6
```

## Finetuning for Document-Level Translation
```bash
python3 main.py --training-mode=finetune \
                --dataset=iwslt17 \
                --data-path=./data \
                --output-path=./output_filename \
                --mem-index= 5 \
                --mem-length=16 \ 
                --mem-side=both \ 
                --mem-pos=sinusoidal \ 
                --batch-size=8 \
                --dropout=0.3 \
                --dropout-mem=0.3 \ 
                --warmup-steps=1000 \ 
                --min-steps=100  \
                --log-steps=100 \
                --eval-steps=500 \
                --learning-rate-sentence=0.00006 \
                --learning-rate-finetune=0.0003  \
                --min-optimize-step=1000 \ # set 1000 for full docmument optimiztion
                --max-optimize-step=1000
```
