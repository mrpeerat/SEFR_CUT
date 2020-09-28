# SEFR CUT
Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble <br>
CRF as Stacked Model and DeepCut as Baseline model<br>

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
- ws1000, tnhc
  - ws1000: Model trained on Wisesight-1000 and test on Wisesight-160
  - tnhc: Model trained on TNHC (80:20 train&test split with random seed 42)
  - BEST: Trained on BEST-2010 Corpus (NECTEC)
  ```
  SEFR_CUT.load_model(engine='ws1000')
  # OR
  SEFR_CUT.load_model(engine='tnhc')
  # OR
  SEFR_CUT.load_model(engine='best')
  ```
- tl-deepcut-XXXX
  - We also provide transfer learning of deepcut on 'Wisesight' as tl-deepcut-ws1000 and 'TNHC' as tl-deepcut-tnhc
  ```
  SEFR_CUT.load_model(engine='tl-deepcut-ws1000')
  # OR
  SEFR_CUT.load_model(engine='tl-deepcut-tnhc')
  ```
- deepcut
  - We also provide the original deepcut
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
<img src="https://user-images.githubusercontent.com/21156980/94403460-aa131980-0197-11eb-9a68-8e2927a5059d.PNG" width="200" height="200" />

## How to re-train?
- .....

## Citation
- ......

Thank you many code from

- [Deepcut](https://github.com/rkcosmos/deepcut) (Baseline Model) : We used 30% code from Deepcut
- [@bact](https://github.com/bact) (CRF training code) : We used 10% from crf_wordseg https://github.com/bact/nlp-thai


