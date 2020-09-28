# SEFR CUT
Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble
DeepCut version (for AttaCut version [here](.........))

## Install

> pip install sefr_cut

## How To use
### Requirements
- python >= 3.6
- python-crfsuite >= 0.9.7
- pyahocorasick == 1.4.0

## Example
You can play on [Notebooks folder](https://github.com/mrpeerat/SEFR_CUT/tree/master/Notebook)
### Load Engine & Engine Mode
- ws1000
  - XXXXXXXXXXXXXX
  ```
  SEFR_CUT.load_model(engine='ws1000')
  ```
- tnhc
  - XXXXXXXXXXXXXX
  ```
  SEFR_CUT.load_model(engine='tnhc')
  ```
- tl-deepcut-XXXX
  - XXXXXXXXXXXXXX
  ```
  SEFR_CUT.load_model(engine='tl-deepcut-xxxx')
  ```
- deepcut
  - XXXXXXXXXXXXXX
  ```
  SEFR_CUT.load_model(engine='deepcut')
  ```
### Segment Example
- Segment with default k
  ```
  SEFR_CUT.load_model(engine='ws1000')
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ']))
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย']))
  print(sefr_cut.tokenize('สวัสดีประเทศไทย'))
  
  [['สวัสดี', 'ประเทศ', 'ไทย'], ['ลุง', 'ตู่', 'สู้', 'ๆ']]
  [['สวัสดี', 'ประเทศ', 'ไทย']]
  [['สวัสดี', 'ประเทศ', 'ไทย']]
  ```
- Segment with different k
  ```
  SEFR_CUT.load_model(engine='ws1000')
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ'],k=5)) # refine only 5% of character number
  print(sefr_cut.tokenize(['สวัสดีประเทศไทย','ลุงตู่สู้ๆ'],k=100)) # refine 100% of character number
  
  [['สวัสดี', 'ประเทศไทย'], ['ลุงตู่', 'สู้', 'ๆ']]
  [['สวัสดี', 'ประเทศ', 'ไทย'], ['ลุง', 'ตู่', 'สู้', 'ๆ']]
  ```
## Performance
- .........

## How to re-train?
- .....

## Citation
- ......

Thank you many code from

- [Deepcut](https://github.com/rkcosmos/deepcut) (Baseline Model) : We used 30% code from Deepcut
- [@bact](https://github.com/bact) (CRF training code) : We used 10% from crf_wordseg https://github.com/bact/nlp-thai


