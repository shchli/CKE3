# KCE3

This is the code of KCE3


### Installation
```
conda create -n tirgn python=3.9

conda activate tirgn

pip install -r requirement.txt
```



## How to run

#### Process data

For all the datasets, the following command can be used to get the history of their entities and relations.
```
cd src
python get_history.py --dataset ICEWS14
```



#### Train models

Then the following commands can be used to train TiRGN.

Train models

```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint
```



#### Evaluate models

The following commands can be used to evaluate TiRGN (add `--test` only).

###### Test with ground truth history:

```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test 
```




