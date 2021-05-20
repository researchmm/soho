import argparse
import os
import pickle
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from apex import amp

import commons
from SOHO.datasets import build_dataloader,build_dataset
from SOHO.models import build_model
from commons.parallel import MMDistributedDataParallel
from commons.runner import get_dist_info,load_checkpoint
from commons.runner import init_dist

def multi_gpu_test(model,data_loader,tmpdir=None):
    model.eval()
    results=[]
    dataset = data_loader.dataset
    rank,world = get_dist_info()
    if rank==0:
        prog_bar = commons.ProgressBar(len(dataset))

    for i,data in enumerate(data_loader):
        with torch.no_grad():
            result = model(mode='test',**data)
        results.append(result)

        if rank==0:
            bs = len(data['img'])
            for _ in range(bs*world):
                prog_bar.update()
    results = collect_results(results, len(dataset), tmpdir)

    return results

def collect_results(result_part, size, tmpdir=None):
    results_out={}
    for k in result_part[0].keys():
        results_out[k] = np.concatenate([batch[k].numpy() for batch in result_part],
                                     axis=0)
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        commons.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    commons.dump(results_out, os.path.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(commons.load(part_file))
        # sort the results
        ordered_results = defaultdict(list)
        out_dict=defaultdict(list)
        for res in part_list:
            for k in part_list[0].keys():
                out_dict[k].append(res[k])

        for k in part_list[0].keys():
            for res in zip(*(out_dict[k])):
                ordered_results[k].extend(list(res))
        # the dataloader may pad some samples
            ordered_results[k] = ordered_results[k][:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description="Test VQA Model")
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', nargs='+', type=int, help='checkpoint file')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi', 'aml'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = commons.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since depends on the dist info
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )
    # build the model and load checkpoint
    model = build_model(cfg.model)
    check_item = args.checkpoint[0]
    checkpoint = load_checkpoint(model, os.path.join(cfg.work_dir, 'epoch_' + str(check_item) + '.pth'),
                                 map_location='cpu')
    label2ans = dataset.label2ans



    gpu_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    model = model.cuda()
    if cfg.fp_16.enable:
        model = amp.initialize(model,
                                          opt_level=cfg.fp_16.opt_level,
                                          loss_scale=cfg.fp_16.loss_scale,
                                          max_loss_scale=cfg.fp_16.max_loss_scale)
        print('**** Initializing mixed precision done. ****')
    model = MMDistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if rank == 0:
        output_path = os.path.join(cfg.work_dir, "test_results")
        commons.mkdir_or_exist(output_path)
        out_list = []
        pickle.dump(outputs, open("outputs.pkl", 'wb'))

        ids = outputs["ids"]
        preds = outputs["pred"]

        for id,pred in zip(ids,preds):
            q_id = dataset.q_id_list[int(id)]
            pred_index = np.argmax(pred, axis=0)
            answer = dataset.label2ans[pred_index]
            out_list.append({'question_id': q_id, 'answer': answer})

        print('\nwriting results to {}'.format(output_path))
        commons.dump(out_list, os.path.join(output_path, "test_submit_{0}.json".format(str(check_item))))
        os.system("rm -rf outputs.pkl")


if __name__ == '__main__':
    main()