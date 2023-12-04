## 1. Summary
### Attention Is All You Need (Transformer) 분석 및 구현 
[논문 분석 노션 링크](https://bit.ly/3Gm5Shx)

### Prepare Dataset
1. raw data: WMT'16 EN-DE Data
2. moses scripts => pre tokenizer, cleaning
3. BPE subwords tokenizer  
#### Result
- `train.tok.clean.bpe.32000.en`: english training input data (4500962)
- `train.tok.clean.bpe.32000.de`: german training output data (4500962)
- `vocab.bpe.32000`: moses tokenizer & BPE tokenizer -> vocabulary (36546)
- `newstest*.tok.clean.bpe.32000.en`: english test input data (22140)
- `newstest*.tok.clean.bpe.32000.de`: german test input data (22140)
#### Reference
- paper: `Massive Exploration of Neural Machine Translation
Architectures`
- docs: `https://google.github.io/seq2seq/nmt/`