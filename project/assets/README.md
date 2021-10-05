# Training assets

The assets folder holds all training and development data for developing the Named Entity Recognition (NER) and Text Classification (Textcat) model. For more information about the annotation guidelines, visit the [healthsea blog post]().

## Named Entity Recognition

<img src="img/ner.svg">

### 5060 examples 
![62.59%](https://progress-bar.dev/63?title=3167) without entities

![37.41%](https://progress-bar.dev/37?title=1893) with entities

| Label | Description | Examples |
| --- | --- | --- | 
| Condition | Diseases, symptoms, problems specified to body region, -organ, and -function | joint pain, diabetes, digestion issues |
| Benefit | Desireable states of health, body region, -organ, and -function | energy, sleep, muscle recovery, skin |

### Label distribution

| Label | Total | Unique | Percentage |
| --- | --- | --- | --- | 
| Condition | 1712 | 715 | ![72%](https://progress-bar.dev/72) | 
| Benefit | 1252 | 282 | ![28%](https://progress-bar.dev/28) | 
|  | 2964 | 997 | ![100%](https://progress-bar.dev/100) | 


## Text Classification

<img src="img/textcat.svg">

### 4979 examples

| Label | Description | Examples |
| --- | --- | --- | 
| Positive | Improvement of health | Increased benefit, decreases condition, improved benefit/condition |
| Negative | Deterioration of health | Decreased benefit, increased condition, made benefit/condition worse |
| Neutral | No effect on health | Not increased benefit, not decreased condition, no effect on benefit/condition |
| Anamnesis | Current state of health | Diagnosed with condition/benefit, suffering from condition/benefit |

### Class distribution

| Class | Total | Percentage |
| --- | --- | --- |
| Positive | 2006 | ![40%](https://progress-bar.dev/40) |
| Negative | 291 | ![6%](https://progress-bar.dev/6) | 
| Neutral | 2268 | ![46%](https://progress-bar.dev/46) | 
| Anamnesis | 414 | ![8%](https://progress-bar.dev/8) |
|  | 4979 | ![100%](https://progress-bar.dev/100) |
