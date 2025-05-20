python producer.py --config config/config.yml --split train
python consumer.py --config config/config.yml --split train
python main.py --config config/config.yml --split test
python trainer.py --config config/config.yml