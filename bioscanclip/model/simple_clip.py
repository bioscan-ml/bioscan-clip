import torch.nn.functional as F
import timm
import torch.nn as nn
from bioscanclip.model.mlp import MLPEncoder
from bioscanclip.model.image_encoder import LoRA_ViT_timm, LoRA_ViT_OpenCLIP
from bioscanclip.model.dna_encoder import load_pre_trained_bioscan_bert, LoRA_barcode_bert, Freeze_DNA_Encoder
from bioscanclip.model.language_encoder import load_pre_trained_bert, LoRA_bert, LoRA_bert_OpenCLIP
from bioscanclip.util.util import add_lora_layer_to_open_clip
import numpy as np
from typing import Optional
import torch
import open_clip


class SimpleCLIP(nn.Module):
    def __init__(self, image_encoder, dna_encoder, language_encoder, open_clip_model=None, init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.language_encoder = language_encoder
        self.open_clip_model = open_clip_model
        self.tokenizer_for_open_clip = open_clip.get_tokenizer('ViT-B-32') if open_clip_model is not None else None
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None


        if self.dna_encoder is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)

        if self.open_clip_model is not None:
            if image_input is not None:
                image_features = self.open_clip_model.encode_image(image_input)
                image_output = F.normalize(image_features, p=2, dim=-1)
            if language_input is not None:
                language_input = self.tokenizer_for_open_clip(language_input, context_length=77)
                language_input = language_input.to(image_input.device)
                text_features = self.open_clip_model.encode_text(language_input)
                language_output = F.normalize(text_features, p=2, dim=-1)
        else:
            if self.image_encoder is not None:
                image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
            if self.language_encoder is not None:
                language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)
        return image_output, dna_output, language_output, self.logit_scale.exp(), self.logit_bias


# Modified from OpenCLIP code
class SimpleCLIP_With_Trainable_Temp(nn.Module):
    def __init__(self, image_encoder, dna_encoder, language_encoder, init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None):
        super(SimpleCLIP_With_Trainable_Temp, self).__init__()
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.language_encoder = language_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None

        if self.image_encoder is not None:
            image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
        if self.dna_encoder is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)
        if self.language_encoder is not None:
            language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)

        return image_output, dna_output, language_output, self.logit_scale.exp(), self.logit_bias


class SimpleCLIPWithClassificationHead(nn.Module):
    def __init__(self, clip_model, hidden_dime=768, number_of_classes=1024):
        super(SimpleCLIPWithClassificationHead, self).__init__()
        self.image_encoder = clip_model.image_encoder
        self.dna_encoder = clip_model.dna_encoder
        self.language_encoder = clip_model.language_encoder

        if self.image_encoder is not None:
            for param in self.image_encoder.parameters():
                param.requires_grad = True
        if self.dna_encoder is not None:
            for param in self.dna_encoder.parameters():
                param.requires_grad = False
        if self.language_encoder is not None:
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(768, hidden_dime),
            nn.ReLU(),
            nn.Linear(hidden_dime, hidden_dime),
            nn.ReLU(),
            nn.Linear(hidden_dime, number_of_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None

        if image_input is not None:
            image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
        if dna_input is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)
        if language_input is not None:
            language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)

        output = self.classification_head(image_output)

        return image_output, dna_output, language_output, output


def load_clip_model(args, device=None):
    torch.cuda.empty_cache()
    # Either we have the small models
    image_encoder = None
    dna_encoder = None
    language_encoder = None

    # Or when we use the OpenCLIP model
    open_clip_model = None

    disable_lora = False
    if hasattr(args.model_config, 'disable_lora'):
        disable_lora = args.model_config.disable_lora

    using_open_clip = False
    if hasattr(args.model_config, 'using_open_clip'):
        disable_lora = args.model_config.using_open_clip

    image_model = None
    if hasattr(args.model_config.image, 'model'):
        image_model = args.model_config.image.model

    language_model = None
    if hasattr(args.model_config.language, 'model'):
        language_model = args.model_config.language.model

    dna_model = None
    if hasattr(args.model_config.dna, 'model'):
        dna_model = args.model_config.dna.model

    if using_open_clip or (image_model == "lora_clip_image" and language_model == "lora_clip_text") :
        open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L/14', pretrained='commonpool_xl_laion_s13b_b90k')
        open_clip_model.to(device)
        if not disable_lora:
            open_clip_model = add_lora_layer_to_open_clip(open_clip_model, r=4, num_classes=args.model_config.output_dim)
    else:
        # For image part
        if args.model_config.image.input_type == "image":
            image_model_name = 'vit_base_patch16_224'
            if hasattr(args.model_config.image, 'pre_train_model'):
                image_model_name = args.model_config.image.pre_train_model
            pre_trained_timm_vit = timm.create_model(image_model_name, pretrained=True)

            # pre_trained_timm_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            if disable_lora:
                image_encoder = LoRA_ViT_timm(vit_model=pre_trained_timm_vit, r=4,
                                              num_classes=args.model_config.output_dim, lora_layer=[])
            else:
                image_encoder = LoRA_ViT_timm(vit_model=pre_trained_timm_vit, r=4,
                                              num_classes=args.model_config.output_dim)
        else:
            image_encoder = MLPEncoder(input_dim=args.model_config.image.input_dim,
                                       hidden_dim=args.model_config.image.hidden_dim,
                                       output_dim=args.model_config.output_dim)

        # For language
        if hasattr(args.model_config, 'language'):
            if args.model_config.language.input_type == "sequence":
                language_model_name = 'prajjwal1/bert-small'
                if hasattr(args.model_config.language, 'pre_train_model'):
                    language_model_name = args.model_config.language.pre_train_model
                _, pre_trained_bert = load_pre_trained_bert(language_model_name)
                if disable_lora:
                    language_encoder = LoRA_bert(model=pre_trained_bert, r=4, num_classes=args.model_config.output_dim,
                                                 lora_layer=[])
                else:
                    language_encoder = LoRA_bert(model=pre_trained_bert, r=4, num_classes=args.model_config.output_dim)
            else:
                raise TypeError(f"Using {args.model_config.language.input_type} as language input is not support yet.")


    # For DNA part
    if hasattr(args.model_config, 'dna'):
        if hasattr(args.model_config.dna, 'freeze') and args.model_config.dna.freeze:
            dna_encoder = Freeze_DNA_Encoder()
        elif args.model_config.dna.input_type == "sequence":
            if dna_model == "barcode_bert" or dna_model == "lora_barcode_bert":
                pre_trained_barcode_bert = load_pre_trained_bioscan_bert(
                    bioscan_bert_checkpoint=args.bioscan_bert_checkpoint)
                if disable_lora:
                    dna_encoder = LoRA_barcode_bert(model=pre_trained_barcode_bert, r=4,
                                                    num_classes=args.model_config.output_dim, lora_layer=[])
                else:
                    dna_encoder = LoRA_barcode_bert(model=pre_trained_barcode_bert, r=4,
                                                    num_classes=args.model_config.output_dim)
        else:
            dna_encoder = MLPEncoder(input_dim=args.model_config.dna.input_dim,
                                     hidden_dim=args.model_config.dna.hidden_dim,
                                     output_dim=args.model_config.output_dim)

    model = SimpleCLIP(image_encoder=image_encoder, dna_encoder=dna_encoder,
                       language_encoder=language_encoder, open_clip_model=open_clip_model)

    if device is not None:
        model.to(device)

    if disable_lora:
        for param in model.parameters():
            param.requires_grad = True

    return model
