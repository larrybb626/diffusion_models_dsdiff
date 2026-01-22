import argparse

import yaml

from Disc_diff.guided_diffusion import dist_util, logger
from Disc_diff.guided_diffusion.image_datasets import load_data, _load_prostate_data, dataset_config
from Disc_diff.guided_diffusion.resample import create_named_schedule_sampler
from Disc_diff.guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser
)
from Disc_diff.guided_diffusion.train_util import TrainLoop


def main():
    # Parse command-line arguments and set up distributed training
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    print(args)
    logger.configure(dir="/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/DiscDiff/BraTS_v_predict_7e4/")

    logger.log("creating model...")
    # Create the super-resolution model and diffusion
    model, diffusion = sr_create_model_and_diffusion(args)
    from pytorch_lightning.utilities import model_summary
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(dist_util.dev())
    # Create the schedule sampler based on the chosen method
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # Load the super-resolution dataset
    # brain_dataset = load_superres_data(
    #     args.hr_data_dir,
    #     args.lr_data_dir,
    #     args.other_data_dir
    # )
    data_config = dataset_config(**args.data_config)
    prostate_dataset = load_prostate_data(data_config)
    logger.log("training...")
    # Start the training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=prostate_dataset,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_superres_data(hr_data_dir, lr_data_dir, other_data_dir):
    # Load the super-resolution data using the specified directories
    return load_data(
        hr_data_dir=hr_data_dir,
        lr_data_dir=lr_data_dir,
        other_data_dir=other_data_dir)


def load_prostate_data(dataset_config):
    # Load the super-resolution data using the specified directories
    return _load_prostate_data(
        dataset_config=dataset_config, )


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file",
                        default="../config/config_train_brats.yaml")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":
    main()
