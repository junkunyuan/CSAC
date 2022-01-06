import argparse,os,torch,numpy,random

def args_parser():
    parser = argparse.ArgumentParser()
    # Output paras.
    parser.add_argument('--outroot', type=str, default="logs",help="the log output dir")
    parser.add_argument('--exp_type', type=str, default='dg',choices=["DG","DA","dg","da"],help="experiment type")

    # Training paras.
    parser.add_argument('--pre-ce', type=str,default='label-smooth', help='the pre-train loss type.')
    parser.add_argument('--epoch', type=int, default=30, help="pre-training epochs")
    parser.add_argument('--length', type=int, default=60, help='iter steps in one epoch')
    parser.add_argument('--round', type=int, default=40, help='round of the training')
    parser.add_argument('--net', type=str, default='resnet18', help='name of net')
    parser.add_argument('--bs', type=int, default=64, help='global batch-size')
    parser.add_argument('--pre_train', action='store_true', help='use public domain to pre-train the model')
    parser.add_argument('--print_step', type=int, default=10, help='record accuracy every print_step in one epoch')
    parser.add_argument('--trace_acc', action='store_true', help='trace acc')
    parser.add_argument('--fix', type=str, default='private', choices=['public', 'private'], help="xxxx")
    parser.add_argument('--mode', type=str, default='alpha', choices=['noa_mmmd', 'noa_mmd', 'all', 'pam', 'cam', 'alpha', 'alpha_cam', 'alpha_pam', 'alpha_all'])
    parser.add_argument('--fusion', type=str, default='l2', choices=['fedavg', 'l1', 'l2', 'cosion'])
    parser.add_argument('--lambda', type=float, default=0.6)
    parser.add_argument('--addition', type=str, default='bone', choices=['bone', 'soft', 'bone_soft'])
    parser.add_argument('--align_loss', type=str, default='mkmmd', choices=['mkmmd', 'mse'])

    # Data paras.
    parser.add_argument('--dataset', type=str, default="pacs", help="dataset name")
    parser.add_argument('--public', type=str, default='A', help="public domain")
    parser.add_argument('--test', type=str, default='*', help='test domain')

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID; -1 for CPU")
    parser.add_argument('--seed', type=int, default=0, help="random seed (default 0)")
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = args.seed
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    return args

def merge_config(args, *yaml_configs):
    config = {}
    for y_config in yaml_configs:
        config.update(y_config)
    config["args"] = args.__dict__

    return config


if __name__ == '__main__':
    args = args_parser()
    yaml_config1 = {
        "dataset":{
            "list_dir":"dafasf",
            "class_num":7
        },
        "process":{
            "scale":(0.5, 0.5)
        }
    }
    config = merge_config(args, yaml_config1)
    print(config)
    
