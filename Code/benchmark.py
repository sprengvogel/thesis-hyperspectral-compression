import argparse
import io
import json
import multiprocessing as mp
import numpy as np
import os
import PIL
import subprocess
import sys
import time
import torch
import math

from collections import defaultdict
from itertools import starmap
from PIL import Image
from compressai.utils.bench.codecs import filesize, read_image
from pytorch_msssim import ms_ssim
from tempfile import mkstemp

from torch_utils import AverageMeter
from torch.utils.data import DataLoader
from hypercomp import params as p
from hypercomp import data

codecs = [
    'JPEG',
    'JPEG2000',
    'WebP'
]


def func(i, *args):
    rv = compress_mat(*args)
    return i, rv


def main(arguments):
    if not os.path.exists(arguments.save_dir):
        os.makedirs(arguments.save_dir)

    # filepaths = os.listdir(arguments.root_dir)
    # filepaths = [os.path.join(arguments.root_dir, path) for path in filepaths]
    test_dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                                 num_workers=p.NUM_WORKERS, pin_memory=False, drop_last=True)
    print(len(test_dataset))
    print(len(test_dataloader))

    for codec in arguments.codecs:
        save_file = f'{arguments.save_dir}/{codec}.json'

        out = defaultdict(dict)
        result = [defaultdict(float)
                  for _ in range(len(arguments.qps))]
        for batch_idx, batch in enumerate(test_dataloader):
            print(f" Batch {batch_idx} started.")
            pool = mp.Pool(
                arguments.num_jobs) if arguments.num_jobs > 1 else None
            args = [(i, f, q, codec) for i, q in enumerate(arguments.qps)
                    for f in batch]

            if pool:
                rv = pool.starmap(func, args)
            else:
                rv = list(starmap(func, args))
            print("Compression and decompression finished.")
            # sum up and average results for all images
            for quality_idx, metrics in rv:
                for k, v in metrics.items():
                    if k == "mse":
                        psnr = compute_psnr(v)
                        if math.isfinite(psnr):
                            result[quality_idx]["psnr"] += psnr / \
                                len(test_dataset)
                        else:
                            print(f"mse:{v}")
                            result[quality_idx]["psnr"] += 50 / \
                                len(test_dataset)
                    else:
                        result[quality_idx][k] += v / len(test_dataset)
            print(f" Batch {batch_idx} finished.")

        # list of dict -> dict of list
        for quality_idx, quality_result in enumerate(result):
            out[str(arguments.qps[quality_idx])] = quality_result

        output = {
            "name": codec,
            "description": f"{codec}. "
                           f"{'Ffmpeg' if codec == 'JPEG2000' else 'Pillow'} version "
                           f"{get_ffmpeg_version() if codec == 'JPEG2000' else get_pillow_version()}",
            "results": out,
        }
        with open(save_file, 'w') as f:
            json.dump(output, f, indent=2)


def compress_mat(img, quality, codec):
    img = img * 255
    img = img.squeeze().detach().cpu().numpy()
    img = img.astype(np.uint8)

    mse_meter = AverageMeter(name="mse meter", length=202)
    # ms_ssim_meter = AverageMeter()
    bpppc_meter = AverageMeter(name="bpppc meter", length=202)
    encoding_time = 0
    decoding_time = 0
    for chan in range(img.shape[0]):
        chan = img[chan, :, :]
        out, rec = compress(chan, quality, codec)
        mse_meter.update(compute_mse(chan, rec))
        # ms_ssim_meter.update(compute_ms_ssim(chan, rec))
        bpppc_meter.update(out['bpppc'])
        encoding_time += out['encoding_time']
        decoding_time += out['decoding_time']
    metrics = {
        "mse": mse_meter.avg,
        "bpppc": bpppc_meter.avg,
        "encoding_time": encoding_time,
        "decoding_time": decoding_time
    }
    return metrics


def compress(chan, quality, codec):
    if codec == 'JPEG':
        return compress_pil(chan, quality, codec)
    elif codec == 'JPEG2000':
        return compress_jpeg2000(chan, quality)
    elif codec == 'WebP':
        return compress_pil(chan, quality, codec)
    else:
        return NotImplementedError()


def compress_pil(chan, quality, codec):
    chan = Image.fromarray(chan)

    start = time.time()
    tmp = io.BytesIO()
    chan.save(tmp, format=codec, quality=int(quality))
    enc_time = time.time() - start

    tmp.seek(0)
    size = tmp.getbuffer().nbytes

    start = time.time()
    rec = Image.open(tmp)
    rec.load()
    dec_time = time.time() - start

    rec = np.asarray(rec)
    if codec == 'WebP':
        rec = rec[:, :, 0]

    bpppc_val = float(size) * 8 / (chan.size[0] * chan.size[1])

    out = {
        'bpppc': bpppc_val,
        'encoding_time': enc_time,
        'decoding_time': dec_time,
    }

    return out, rec


def compress_jpeg2000(chan, quality):
    fd0, in_filepath = mkstemp(suffix=".png")
    fd1, out_filepath = mkstemp(suffix=".jp2")
    fd2, rec_filepath = mkstemp(suffix=".png")

    chan = Image.fromarray(chan)
    chan.save(in_filepath, "PNG")

    encode_cmd = [
        "/usr/bin/ffmpeg",
        "-loglevel",
        "panic",
        "-y",
        "-i",
        in_filepath,
        "-pix_fmt",
        "gray",
        "-c:v",
        "libopenjpeg",
        "-compression_level",
        quality,
        out_filepath
    ]

    decode_cmd = [
        "/usr/bin/ffmpeg",
        "-loglevel",
        "panic",
        "-y",
        "-i",
        out_filepath,
        rec_filepath
    ]

    # Encode
    start = time.time()
    run_command(encode_cmd)
    enc_time = time.time() - start

    size = filesize(out_filepath)

    # Decode
    start = time.time()
    run_command(decode_cmd)
    dec_time = time.time() - start

    # Read image
    rec = read_image(rec_filepath, mode="L")
    os.close(fd1)
    os.remove(out_filepath)
    os.close(fd2)
    os.remove(rec_filepath)

    chan = read_image(in_filepath, mode="L")
    os.close(fd0)
    os.remove(in_filepath)
    bpppc_val = float(size) * 8 / (chan.size[0] * chan.size[1])

    out = {
        'bpppc': bpppc_val,
        'encoding_time': enc_time,
        'decoding_time': dec_time,
    }

    return out, rec


def compute_mse(a, b):
    mse = ((a - b) ** 2).mean()
    return mse


def compute_psnr(mse, max_val=255.0):
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def compute_ssim(a, b):
    return NotImplementedError()


def compute_sam(a, b):
    return NotImplementedError()


def compute_ms_ssim(a, b, max_val=255.0):
    a = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    b = torch.from_numpy(b).unsqueeze(0).unsqueeze(0)
    return ms_ssim(a, b, data_range=max_val)


def run_command(cmd):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        print(err.output.decode("utf-8"))
        sys.exit(1)


def get_ffmpeg_version():
    rv = run_command(["ffmpeg", "-version"])
    return rv.split()[2]


def get_pillow_version():
    return PIL.__version__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create benchmark results for standard codecs')
    parser.add_argument('--save-dir', default='./results/baseline/', type=str,
                        help='path to the output directory')
    parser.add_argument(
        "-j",
        "--num-jobs",
        type=int,
        metavar="N",
        default=1,
        help="number of parallel jobs (default: %(default)s)",
    )
    parser.add_argument(
        '--codecs',
        default=['JPEG', 'JPEG2000', 'WebP'],
        nargs="+",
        type=str,
        choices=codecs,
        help='codecs to evaluate')
    parser.add_argument(
        "-q",
        "--qps",
        dest="qps",
        metavar="Q",
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45,
                 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        nargs="+",
        type=int,
        help="list of quality/quantization parameter (default: %(default)s)",
    )

    parse_args = parser.parse_args()
    main(parse_args)
