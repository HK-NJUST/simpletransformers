# sudo wget https://www.python.org/ftp/python/3.8.19/Python-3.8.19.tgz
# sudo tar -xzf Python-3.8.19.tgz
# cd Python-3.8.19
# sudo ./configure --enable-optimizations
# sudo make altinstall
# python3.8 --version
# pip3.8 --version
ls /home/work/datasets/
# ls /home/work/datasets/train02
# ls /home/work/datasets/train02/train_240801_240810
pip3 install -r requirements-dev.txt --user
pip3 list


# 单卡A100
# torchrun \
#     --nproc_per_node 1 \
#     train_multi_modal_model_cls.py

# 多卡A100
torchrun \
    --nproc_per_node 1 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_multi_modal_model_cls_add_lr.py