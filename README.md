This repository contains a project on counting number of people in crowd using an improved version of a [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) model. The model uses VGG16 backbone to extract features and conducts classification, regression, upsampling based on the extracted features. The model parts can be modified in models/p2pnet.py file.

### Virtual Environment Creation

```python

conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt

```

### Environment to run the code in 218 server

```python

conda activate imagen

```

### Training process

```
python train.py --data_root /path/to/dataset --dataset_file JHU --gpu_id 1
```

### Inference

```
python run_test.py --weight_path /path/to/trained/model --output_dir /path/to/save/results 
```

