# NLP Homework 3

## Run
```shell script
python main.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500 --model_type=GAT
python main.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500 --model_type=GraphSage
python main.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500 --model_type=GCN


python main.py --dataset=citeseer --weight_decay=5e-3 --num_layers=3 --epochs=500 --model_type=GCN
python main.py --dataset=citeseer --weight_decay=5e-3 --num_layers=3 --epochs=500 --model_type=GraphSage
python main.py --dataset=citeseer --weight_decay=5e-3 --num_layers=3 --epochs=500 --model_type=GAT

```



