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

Some inference results can be seen below.

![Picture2](https://user-images.githubusercontent.com/50166164/236081813-ff21dcea-d952-4e57-bc76-60021f5d25a4.png)
![Picture3](https://user-images.githubusercontent.com/50166164/236081829-906283e9-83e4-4194-9d53-842e09edef8a.png)
![Picture4](https://user-images.githubusercontent.com/50166164/236081980-01bdb7bd-b9ca-43f3-adae-7bc62ccc8ff5.png)
![Picture5](https://user-images.githubusercontent.com/50166164/236081999-df7dd247-dec5-44f5-8596-eddc1d0e5851.png)
![Picture6](https://user-images.githubusercontent.com/50166164/236082010-cdfefd92-8cb7-4052-9cc6-02c1b28f4832.png)
![Picture7](https://user-images.githubusercontent.com/50166164/236082018-bedf0e28-c333-4488-ad1e-133eb5b9db49.png)
![Picture8](https://user-images.githubusercontent.com/50166164/236082037-796e79fb-d3a6-423d-b5bd-fc0b7d6762c5.png)

