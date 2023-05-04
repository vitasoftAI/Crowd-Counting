This repository contains a project on counting number of people in crowd using an improved version of a [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) model. The model uses VGG16 backbone to extract features and conducts classification, regression, upsampling based on the extracted features. The model parts can be modified in models/p2pnet.py file.

### Virtual Environment Creation

```python

conda create -n <ENV_NAME> python = 3.9
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

Generated sample LPs can be seen below:

![Picture2](https://user-images.githubusercontent.com/50166164/236081813-ff21dcea-d952-4e57-bc76-60021f5d25a4.png)
![Picture3](https://user-images.githubusercontent.com/50166164/236081829-906283e9-83e4-4194-9d53-842e09edef8a.png)



![Picture9](https://user-images.githubusercontent.com/50166164/236081777-0a64c138-b382-4310-9fd5-403cc28a29a1.png)

