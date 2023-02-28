# example-classifier

## Prepare dataset
```
gender/
├── train
│   ├── female
│   └── male
└── val
    ├── female
    └── male
```

The training dataset contains 6500 female and 6500 male.  
The val dataset contains 500 female and 500 male.

## Train
```
python train.py --dataet gender --epochs 100 --batch-size 128 --height 256 --width 128
```

## Val
| net | female | male |
| ------ | ------ | ------ |
| resnet50 | 80.4 % | 87.6 % |
