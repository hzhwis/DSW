import argparse
import torch.utils.checkpoint
from accelerate.logging import get_logger
from torchvision import transforms
from diffusers.utils import check_min_version
import time
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import utils.transformed as trans
from data.ImageFolderDataset import MyImageFolder
from data.ImageFolderDataset_caption import MyImageFolder_captions
from models.HidingUNet import UnetGenerator
from models.HidingRes import HidingRes
import numpy as np
from vgg import Vgg16
import torch
from diffusers import StableDiffusionPipeline
import shutil, os
from PIL import Image
import pytorch_ssim
import torch
from torch.autograd import Variable
from get_files_list import get_my_files_list, get_my_files_prompts

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def set_path(experiment_dir_0, batchSize, resume_time, if_resume):
    if if_resume:
        cur_time = resume_time
    else:
        cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = experiment_dir_0 + cur_time
    outckpts = experiment_dir + "/checkPoints"
    trainpics = experiment_dir + "/trainPics"
    validationpics = experiment_dir + "/validationPics"
    outlogs = experiment_dir + "/trainingLogs"
    testPics = experiment_dir + "/testPics"
    if not if_resume:
        if not os.path.exists(outckpts):
            os.makedirs(outckpts)
        if not os.path.exists(trainpics):
            os.makedirs(trainpics)
        if not os.path.exists(validationpics):
            os.makedirs(validationpics)
        if not os.path.exists(outlogs):
            os.makedirs(outlogs)
        if not os.path.exists(testPics):
            os.makedirs(testPics)
    logPath = outlogs + '/%s_%d_log.txt' % ('train', batchSize)
    return outckpts, trainpics, validationpics, outlogs, outcodes, testPics, runfolder, logPath

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # load model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stable-diffusion/",
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    # loda dataset
    parser.add_argument("--dataset_name", type=str, default='',)
    parser.add_argument("--root_dataset_dir", type=str, default='', )
    parser.add_argument("--clean_gen_data_dir", type=str, default='')
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--secret", type=str, default='')
    parser.add_argument("--version", type=str, default='')
    parser.add_argument("--clean_image", type=str, default='secret/clean.png')
    parser.add_argument("--secret_image", type=str, default='secret/copyright.png')
    parser.add_argument("--select_train", type=str, default='')
    parser.add_argument("--select_test", type=str, default='')
    parser.add_argument("--train_data_dir", type=str, default='')
    parser.add_argument("--test_data_dir", type=str, default='')
    parser.add_argument("--resume_Hnet", type=str, default='')
    parser.add_argument("--resume_Rnet", type=str, default='')

    #output
    parser.add_argument("--root_model_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='copyright_color3/train/lora_8.0/')
    parser.add_argument("--save_train_data_dir", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default='cache/')
    parser.add_argument("--max_train_samples", type=int, default=None)

    ### PRE
    parser.add_argument("--output_dir_0", type=str,
                        default='')
    parser.add_argument("--save_mid_lora_train_data_dir_0", type=str,
                        default="")
    parser.add_argument("--save_mid_lora_test_data_dir_0", type=str,
                        default="")
    parser.add_argument("--experiment_dir_0", type=str, default='')


    # training details
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--imageSize", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_column", type=str, default="image",
                        help="The column of the dataset containing an image.")
    parser.add_argument("--caption_column", type=str, default="text",
                        help="The column of the dataset containing a caption or a list of captions.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--center_crop", default=False, action="store_true",
                        help=("Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                            " cropped. The images will be resized to the resolution first before cropping."))
    parser.add_argument("--random_flip", action="store_true",
                        help="whether to randomly flip images horizontally")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale the learning rate by the num ber of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'))
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--snr_gamma", type=float, default=None,
                        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
                             "More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"))
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."))
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--prediction_type", type=str, default=None,
                        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."))
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."))
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'))
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
                            " training using `--resume_from_checkpoint`."))
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                        help=("Max number of checkpoints to store."))

    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--rank", type=int, default=4, help=("The dimension of the LoRA update matrices."))

    # validation settings
    parser.add_argument("--validation_prompt", type=str, default='', help="A prompt that is sampled during training for inference.")
    parser.add_argument("--num_validation_images", type=int, default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_epochs", type=int, default=1,
        help=("Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."))

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())

    args.clean_gen_data_dir = args.root_dataset_dir + args.clean_gen_data_dir
    args.train_data_dir = args.root_dataset_dir + 'training_data/' + args.select_train
    args.test_data_dir = args.root_dataset_dir + 'training_data/' + args.select_test
    args.cache_dir = args.root_dataset_dir + args.cache_dir
    args.save_train_data_dir = args.root_dataset_dir + 'DMW+lora/' + args.secret + '/mid_results/' + args.version + '/'

    args.output_dir = args.root_model_dir + args.secret + '/train/' + args.version + '/'

    train_prompts = get_my_files_prompts(args.train_data_dir)
    train_list = get_my_files_list(args.train_data_dir)
    test_prompts = get_my_files_prompts(args.test_data_dir)
    test_list = get_my_files_list(args.test_data_dir)

    outckpts, trainpics, validationpics, outlogs, outcodes, testPics, runfolder, logPath = \
            set_path(args.experiment_dir_0, args.train_batch_size, args.resume_time, args.if_resume)
    down_num = 5
    lr = 0.0002
    beta1 = 0.5
    betamse = 10000
    betacons = betacleanA = betacleanB = beta = 1
    betaclean = 1
    betawatermark = 1
    betavgg = 1
    betapix = 1
    betssim = 1
    betagans = 0.01

    train_dir = args.train_data_dir
    train_dataset = MyImageFolder_captions(
        train_dir,
        args.clean_gen_data_dir,
        trans.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            trans.ToTensor(),
        ]))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

    val_dir = args.test_data_dir
    val_dataset = MyImageFolder_captions(
        val_dir,
        args.clean_gen_data_dir,
        trans.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            trans.ToTensor(),
        ]))
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

    Hnet = UnetGenerator(input_nc=6, output_nc=3,  num_downs=down_num, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)
    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.cuda()
    Rnet.apply(weights_init)

    args.patch = (1, args.resolution // 2 ** 4, args.resolution // 2 ** 4)
    optimizerH = optim.Adam(Hnet.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)
    optimizerR = optim.Adam(Rnet.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

    if args.if_resume:
        Hnet.load_state_dict(torch.load(args.resume_Hnet))
        Rnet.load_state_dict(torch.load(args.resume_Rnet))

    # define loss
    mse_loss = nn.MSELoss().cuda()
    criterion_GAN = nn.MSELoss().cuda()
    criterion_pixelwise = nn.L1Loss().cuda()
    vgg = Vgg16(requires_grad=False).cuda()
    smallestLoss = 1000000

    start_epoch = 0
    if args.if_resume:
        start_epoch = args.resume_epoch
    for epoch in range(start_epoch, args.num_train_epochs):
        print('\n###############################  Epoch %i  ###############################' % (epoch))
        cur_output_dir = args.output_dir + cur_time + '/'
        tmp = args.save_train_data_dir + cur_time + '/'
        cur_save_train_data_dir = tmp + 'train/' + str(epoch) + '/'
        bef_save_train_data_dir = tmp + 'train/' + str(epoch-1) + '/'
        cur_save_mid_lora_train_data_dir = tmp + 'lora/' + args.select_train + str(epoch) + '/'
        cur_save_mid_lora_test_data_dir = tmp + 'lora/' + args.select_test + str(epoch) + '/'
        if not os.path.exists(cur_save_train_data_dir):
            os.makedirs(cur_save_train_data_dir)
        if not os.path.exists(cur_save_mid_lora_train_data_dir):
            os.makedirs(cur_save_mid_lora_train_data_dir)
        if not os.path.exists(cur_save_mid_lora_test_data_dir):
            os.makedirs(cur_save_mid_lora_test_data_dir)

        if epoch == 0:
            cur_output_dir = args.output_dir_0
            cur_save_mid_lora_train_data_dir = args.save_mid_lora_train_data_dir_0
            cur_save_mid_lora_test_data_dir = args.save_mid_lora_test_data_dir_0
        else:
            train_lora(args.dataset_name, cur_output_dir, bef_save_train_data_dir, 30)

            pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion/", torch_dtype=torch.float16)
            pipeline.unet.load_attn_procs(cur_output_dir)
            pipeline.to("cuda")
            pipeline.set_progress_bar_config(disable=True)

            generator = torch.Generator("cuda")
            if args.seed != None:
                generator = generator.manual_seed(args.seed)
            for ii in range(len(train_prompts)):
                cur_caption = train_prompts[ii]
                gen_image_ = pipeline(cur_caption, num_inference_steps=30, generator=generator).images[0]
                seed_ = 666
                while is_image_all_black(gen_image_):
                    seed_ += 1
                    generator = torch.Generator("cuda").manual_seed(seed_)
                    gen_image_ = pipeline(cur_caption, num_inference_steps=30, generator=generator).images[0]
                gen_image_.save(cur_save_mid_lora_train_data_dir + train_list[ii])
            for ii in range(len(test_prompts)):
                cur_caption = test_prompts[ii]
                gen_image_ = pipeline(cur_caption, num_inference_steps=30, generator=generator).images[0]
                seed_ = 666
                while is_image_all_black(gen_image_):
                    seed_ += 1
                    generator = torch.Generator("cuda").manual_seed(seed_)
                    gen_image_ = pipeline(cur_caption, num_inference_steps=30, generator=generator).images[0]
                gen_image_.save(cur_save_mid_lora_test_data_dir + test_list[ii])

        Hlosses_t = AverageMeter()
        Rlosses_t = AverageMeter()
        R_mselosses_t = AverageMeter()
        Ganlosses_t = AverageMeter()
        Pixellosses_t = AverageMeter()
        Vgglosses_t = AverageMeter()
        SumLosses_t = AverageMeter()
        Ssimlosses_t = AverageMeter()
        VggAlosses_t = AverageMeter()
        VggBlosses_t = AverageMeter()

        Hnet.train()
        Rnet.train()
        # Tensor type
        Tensor = torch.cuda.FloatTensor
        loader = trans.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), ])
        clean_img_ori = Image.open(args.clean_image).convert('RGB')
        clean_img_ori = loader(clean_img_ori)
        secret_img_ori = Image.open(args.secret_image).convert('RGB')
        secret_img_ori = loader(secret_img_ori)

        for step, (batch, clean_batch, captions_) in enumerate(train_loader, 0):
            '''##################################  original --> Hnet  ##################################'''
            Hnet.zero_grad()
            Rnet.zero_grad()
            this_batch_size = int(batch.size()[0])
            org_img = batch[0:this_batch_size, :, :, :]
            org_gen_img = clean_batch[0:this_batch_size, :, :, :]
            org_A = org_img[:, :, 0:args.resolution, 0:args.resolution]
            org_B = org_img[:, :, 0:args.resolution, 0:args.resolution]
            org_gen_img_A = org_gen_img[:, :, 0:args.resolution, 0:args.resolution]
            secret_img = secret_img_ori.repeat(this_batch_size, 1, 1, 1)
            secret_img = secret_img[0:this_batch_size, :, :, :]
            clean_img = clean_img_ori.repeat(this_batch_size, 1, 1, 1)
            clean_img = clean_img[0:this_batch_size, :, :, :]
            org_A = org_A.cuda()
            org_B = org_B.cuda()
            org_gen_img_A = org_gen_img_A.cuda()
            secret_img = secret_img.cuda()
            clean_img = clean_img.cuda()
            concat_B_secret = torch.cat([org_B, secret_img], dim=1)
            concat_imgv = Variable(concat_B_secret)
            B_imgv = Variable(org_B)
            A_org_gen_imgv = Variable(org_gen_img_A)
            generator_H_img = Hnet(concat_imgv)
            A_imgv = Variable(org_A)
            valid = Variable(Tensor(np.ones((B_imgv.size(0), *args.patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((B_imgv.size(0), *args.patch))), requires_grad=False)

            if this_batch_size % 2 != 1:
                cur_image_0 = Image.open(cur_save_mid_lora_train_data_dir + captions_[0])
                cur_image_1 = Image.open(cur_save_mid_lora_train_data_dir + captions_[1])
                cur_image_0 = loader(cur_image_0)
                cur_image_1 = loader(cur_image_1)
                generator_L_img = torch.stack((cur_image_0, cur_image_1)).cuda()
            else:
                cur_image_0 = Image.open(cur_save_mid_lora_train_data_dir + captions_[0])
                cur_image_0 = loader(cur_image_0)
                generator_L_img = cur_image_0.cuda()

            ssim_loss = 1 - pytorch_ssim.ssim(generator_H_img, B_imgv)
            pixel_loss = criterion_pixelwise(generator_H_img, B_imgv)  # l1
            generator_H_img_rgb = generator_H_img.repeat(1, 1, 1, 1)
            B_imgv_rgb = B_imgv.repeat(1, 1, 1, 1)
            B_imgv_rgb.detach()
            vgg_loss = mse_loss(vgg(generator_H_img_rgb).relu2_2, vgg(B_imgv_rgb).relu2_2)
            errH = betamse * mse_loss(generator_H_img, B_imgv) + betssim * ssim_loss + betapix * pixel_loss + betavgg * vgg_loss

            R_L_img = Rnet(generator_L_img)
            secret_imgv = Variable(secret_img)

            errR_mse = betamse * betawatermark * mse_loss(R_L_img, secret_imgv) + \
                       (1 - pytorch_ssim.ssim(R_L_img, secret_imgv)) + criterion_pixelwise(R_L_img, secret_imgv)
            R_img_A = Rnet(A_imgv)
            clean_imgv = Variable(clean_img)
            clean_imgv_rgb = clean_imgv.repeat(1, 1, 1, 1)
            R_img_A_rgb = R_img_A.repeat(1, 1, 1, 1)
            vgg_A_loss = mse_loss(vgg(R_img_A_rgb).relu2_2, vgg(clean_imgv_rgb).relu2_2)
            errR_clean_A = betamse * mse_loss(R_img_A, clean_imgv) + betavgg * vgg_A_loss
            R_img_B = Rnet(A_org_gen_imgv)
            clean_imgv = Variable(clean_img)
            R_img_B_rgb = R_img_B.repeat(1, 1, 1, 1)
            vgg_B_loss = mse_loss(vgg(R_img_B_rgb).relu2_2, vgg(clean_imgv_rgb).relu2_2)
            errR_clean_B = betamse * mse_loss(R_img_B, clean_imgv) + betavgg * vgg_B_loss
            errR_clean = betacleanA * errR_clean_A + betacleanB * errR_clean_B
            errR = errR_mse + betaclean * errR_clean
            betaerrR_secret = beta * errR
            err_sum = errH + betaerrR_secret
            err_sum.backward()
            optimizerH.step()
            optimizerR.step()

            Hlosses_t.update(errH.data, this_batch_size)
            Rlosses_t.update(errR.data, this_batch_size)
            R_mselosses_t.update(errR_mse.data, this_batch_size)
            Ganlosses_t.update(gan_loss.data, this_batch_size)
            Pixellosses_t.update(pixel_loss.data, this_batch_size)
            VggAlosses_t.update(vgg_A_loss.data, this_batch_size)
            VggBlosses_t.update(vgg_B_loss.data, this_batch_size)
            Vgglosses_t.update(vgg_loss.data, this_batch_size)
            Ssimlosses_t.update(ssim_loss.data, this_batch_size)
            SumLosses_t.update(err_sum.data, this_batch_size)

        Hnet.eval()
        Rnet.eval()
        Hlosses_v = AverageMeter()
        Rlosses_v = AverageMeter()
        R_mselosses_v = AverageMeter()
        R_consistlosses_v = AverageMeter()
        Ganlosses_v = AverageMeter()
        Pixellosses_v = AverageMeter()
        Vgglosses_v = AverageMeter()
        Ssimlosses_t = AverageMeter()
        VggAlosses_t = AverageMeter()
        VggBlosses_t = AverageMeter()
        Tensor = torch.cuda.FloatTensor

        with torch.no_grad():
            clean_img_ori = Image.open(args.clean_image).convert('RGB')
            clean_img_ori = loader(clean_img_ori)
            secret_img_ori = Image.open(args.secret_image).convert('RGB')
            secret_img_ori = loader(secret_img_ori)

            for step, (batch, clean_batch, captions_) in enumerate(val_loader, 0):
                Hnet.zero_grad()
                Rnet.zero_grad()
                this_batch_size = int(batch.size()[0])
                org_img = batch[0:this_batch_size, :, :, :]
                org_gen_img = clean_batch[0:this_batch_size, :, :, :]
                org_A = org_img[:, :, 0:args.resolution, 0:args.resolution]
                org_B = org_img[:, :, 0:args.resolution, 0:args.resolution]
                org_gen_img_A = org_gen_img[:, :, 0:args.resolution, 0:args.resolution]
                secret_img = secret_img_ori.repeat(this_batch_size, 1, 1, 1)
                secret_img = secret_img[0:this_batch_size, :, :, :]
                clean_img = clean_img_ori.repeat(this_batch_size, 1, 1, 1)
                clean_img = clean_img[0:this_batch_size, :, :, :]
                org_A = org_A.cuda()
                org_B = org_B.cuda()
                org_gen_img_A = org_gen_img_A.cuda()
                secret_img = secret_img.cuda()
                clean_img = clean_img.cuda()
                concat_B_secret = torch.cat([org_B, secret_img], dim=1)
                concat_imgv = Variable(concat_B_secret)
                B_imgv = Variable(org_B)
                A_org_gen_imgv = Variable(org_gen_img_A)

                generator_H_img = Hnet(concat_imgv)
                A_imgv = Variable(org_A)
                valid = Variable(Tensor(np.ones((B_imgv.size(0), *args.patch))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((B_imgv.size(0), *args.patch))), requires_grad=False)

                if this_batch_size % 2 != 1:
                    cur_image_0 = Image.open(cur_save_mid_lora_test_data_dir + captions_[0])
                    cur_image_1 = Image.open(cur_save_mid_lora_test_data_dir + captions_[1])
                    cur_image_0 = loader(cur_image_0)
                    cur_image_1 = loader(cur_image_1)
                    generator_L_img = torch.stack((cur_image_0, cur_image_1)).cuda()
                else:
                    cur_image_0 = Image.open(cur_save_mid_lora_test_data_dir + captions_[0])
                    cur_image_0 = loader(cur_image_0)
                    generator_L_img = cur_image_0.cuda()

                ssim_loss = 1 - pytorch_ssim.ssim(generator_H_img, B_imgv)
                pixel_loss = criterion_pixelwise(generator_H_img, B_imgv)
                generator_H_img_rgb = generator_H_img.repeat(1, 1, 1, 1)
                B_imgv_rgb = B_imgv.repeat(1, 1, 1, 1)
                B_imgv_rgb.detach()
                vgg_loss = mse_loss(vgg(generator_H_img_rgb).relu2_2, vgg(B_imgv_rgb).relu2_2)
                errH = betamse * mse_loss(generator_H_img, B_imgv) + betssim * ssim_loss + betapix * pixel_loss + betavgg * vgg_loss

                R_L_img = Rnet(generator_L_img)
                secret_imgv = Variable(secret_img)
                errR_mse = betamse * betawatermark * mse_loss(R_L_img, secret_imgv) + \
                           (1 - pytorch_ssim.ssim(R_L_img, secret_imgv)) + criterion_pixelwise(R_L_img, secret_imgv)
                R_img_A = Rnet(A_imgv)
                clean_imgv = Variable(clean_img)
                clean_imgv_rgb = clean_imgv.repeat(1, 1, 1, 1)
                R_img_A_rgb = R_img_A.repeat(1, 1, 1, 1)
                vgg_A_loss = mse_loss(vgg(R_img_A_rgb).relu2_2, vgg(clean_imgv_rgb).relu2_2)
                errR_clean_A = betamse * mse_loss(R_img_A, clean_imgv) + betavgg * vgg_A_loss
                R_img_B = Rnet(A_org_gen_imgv)
                clean_imgv = Variable(clean_img)
                R_img_B_rgb = R_img_B.repeat(1, 1, 1, 1)
                vgg_B_loss = mse_loss(vgg(R_img_B_rgb).relu2_2, vgg(clean_imgv_rgb).relu2_2)
                errR_clean_B = betamse * mse_loss(R_img_B, clean_imgv) + betavgg * vgg_B_loss
                errR_clean = betacleanA * errR_clean_A + betacleanB * errR_clean_B
                errR = errR_mse + betaclean * errR_clean

                Hlosses_v.update(errH.data, this_batch_size)
                Rlosses_v.update(errR.data, this_batch_size)
                R_mselosses_v.update(errR_mse.data, this_batch_size)
                Ganlosses_v.update(gan_loss.data, this_batch_size)
                Pixellosses_v.update(pixel_loss.data, this_batch_size)
                VggAlosses_t.update(vgg_A_loss.data, this_batch_size)
                VggBlosses_t.update(vgg_B_loss.data, this_batch_size)
                Vgglosses_t.update(vgg_loss.data, this_batch_size)
                Ssimlosses_t.update(ssim_loss.data, this_batch_size)

        val_hloss = Hlosses_v.avg
        val_rloss = Rlosses_v.avg
        val_r_mseloss = R_mselosses_v.avg
        val_r_consistloss = R_consistlosses_v.avg
        val_Ganlosses = Ganlosses_v.avg
        val_Pixellosses = Pixellosses_v.avg
        val_Vgglosses = Vgglosses_v.avg
        val_sumloss = val_hloss + beta * val_rloss

        schedulerH.step(val_sumloss)
        schedulerR.step(val_rloss)
        if val_sumloss < smallestLoss:
            smallestLoss = val_sumloss
            torch.save(Hnet.state_dict(),
                       '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (outckpts, epoch, val_sumloss, val_hloss))
            torch.save(Rnet.state_dict(),
                       '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (outckpts, epoch, val_sumloss, val_rloss))

        train_dir_0 = args.train_data_dir
        train_dataset_0 = MyImageFolder_captions(
            train_dir_0,
            args.clean_gen_data_dir,
            trans.Compose([
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                trans.ToTensor(),
            ]))
        train_loader_0 = DataLoader(train_dataset_0, batch_size=args.train_batch_size, shuffle=False, num_workers=8)
        for step_0, (batch_0, clean_batch_0, captions_0) in enumerate(train_loader_0, 0):
            this_batch_size = int(batch_0.size()[0])
            org_img = batch_0[0:this_batch_size, :, :, :]
            org_B = org_img[:, :, 0:args.resolution, 0:args.resolution]
            org_B = org_B.cuda()

            secret_img_0 = Image.open(args.secret_image).convert('RGB')
            secret_img_0 = loader(secret_img_0)
            secret_img_0 = secret_img_0.repeat(this_batch_size, 1, 1, 1)
            secret_img_0 = secret_img_0[0:this_batch_size, :, :, :]
            secret_img_0 = secret_img_0.cuda()
            concat_B_secret = torch.cat([org_B, secret_img_0], dim=1)
            concat_imgv = Variable(concat_B_secret)
            generator_H_img_0 = Hnet(concat_imgv)
            if this_batch_size == 1:
                save_output = generator_H_img_0.data[0].resize_(1, 3, args.imageSize, args.imageSize)
                cur_save_dir = cur_save_train_data_dir + captions_0[0]
                vutils.save_image(save_output, cur_save_dir, nrow=1, padding=1, normalize=False)
            elif this_batch_size == 2:
                save_output_0 = generator_H_img_0.data[0].resize_(1, 3, args.imageSize, args.imageSize)
                save_output_1 = generator_H_img_0.data[1].resize_(1, 3, args.imageSize, args.imageSize)
                cur_save_dir_0 = cur_save_train_data_dir + captions_0[0]
                cur_save_dir_1 = cur_save_train_data_dir + captions_0[1]
                vutils.save_image(save_output_0, cur_save_dir_0, nrow=1, padding=1, normalize=False)
                vutils.save_image(save_output_1, cur_save_dir_1, nrow=1, padding=1, normalize=False)


def is_image_all_black(img):
    try:
        img_gray = img.convert('L')
        pixels = list(img_gray.getdata())

        if all(pixel == 0 for pixel in pixels):
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    main()
