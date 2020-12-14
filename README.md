# Relation Classification
This project's main objective is learning about how do different classifiers
react to different proportions' between negative and positive relations.

## Script Usage
```shell script
sudo apt install virtualenv && python3-pip
virtualenv --python=/usr/bin/python3.6 venv/
source venv/bin/activate
pip3 install -r requirements.txt
python3.6 setup.py install
python bin/clasification-mineria.py -t resources/bacteria_biotope/train/ -d resources/bacteria_biotope/dev/ -o out/
```

