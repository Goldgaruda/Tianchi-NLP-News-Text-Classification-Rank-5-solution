# Tianchi-NLP-News-Text-Classification-Rank-5-solution
Alibaba Cloud TIANCHI NLP Competition
The introduction of the activity and whole dataset are available: https://tianchi.aliyun.com/competition/entrance/531810/introduction

The raw text has been encrypted by organizer and was finally anonymous so that characters were expressed by a series of numbers. Starters are encouraged to go through the whole competition to have a better understanding of NLP and its framework.

env:
pytorch 1.6.0
sklearn
tensorflow 1.14.0 (for Pre-Train)
lightgbm
tqdm 4.47.*
transformers 3.0.*
Dataset

The news text was divided into 14 classes and labeled so that it's typical supervised learning. Training set and public testing set were avalible before final round, the ranking would be based on the performace of private testing set.
