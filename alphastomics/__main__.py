"""
支持 python -m alphastomics.main 方式运行

用法:
    # 单机单卡
    python -m alphastomics.main train --config config.yaml
    
    # 单机多卡 (torchrun)
    torchrun --nproc_per_node=4 -m alphastomics.main train --config config.yaml
    
    # 多机多卡 (torchrun)
    # 在节点 0 (master):
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
        --master_addr=<master_ip> --master_port=29500 \
        -m alphastomics.main train --config config.yaml
        
    # 在节点 1:
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
        --master_addr=<master_ip> --master_port=29500 \
        -m alphastomics.main train --config config.yaml
"""

from alphastomics.main import main

if __name__ == "__main__":
    main()
