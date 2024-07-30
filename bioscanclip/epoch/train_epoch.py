from tqdm import tqdm
import wandb
import psutil
import torch.distributed.nn
import torch.distributed as dist





def train_epoch(activate_wandb, total_epochs, epoch, dataloader, model, optimizer, criterion, device, open_clip_ver=False, rank=None, check_cuda_memory=False):
    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    epoch_loss = 0.0
    total_step = len(dataloader)

    model.train()
    for step, batch in pbar:
        processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_for_train_batch = batch
        language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                          'attention_mask': attention_mask.to(device)}

        optimizer.zero_grad()
        image_input_batch = image_input_batch.to(device)
        dna_input_batch = dna_input_batch.to(device)
        logit_scale = None
        logit_bias = None
        if open_clip_ver:
            image_output, dna_output, language_output, logit_scale, logit_bias = model(image_input_batch, dna_input_batch,
                                                          language_input)
        else:
            image_output, dna_output, language_output = model(image_input_batch, dna_input_batch, language_input)

        label_for_train_batch = label_for_train_batch.to(device)
        # label_for_train_batch = construct_label_metrix(label_for_train_batch).to(device)

        if open_clip_ver:
            loss = criterion(image_output, dna_output, language_output, label_for_train_batch, logit_scale, logit_bias)
        else:
            loss = criterion(image_output, dna_output, language_output, label_for_train_batch)

        loss.backward()
        print(model.image_encoder.lora_vit.blocks[0].attn.qkv.linear_a_q.weight.grad)
        exit()
        epoch_loss = epoch_loss + loss.item()


        optimizer.step()
        memory_info = torch.cuda.memory_stats(device=device)
        allocated_memory = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(device).total_memory
        cached_memory = torch.cuda.memory_reserved(device)

        total_used_memory = allocated_memory + cached_memory
        if rank == 0:
            pbar.set_description(
                f'Epoch: {epoch}||Step: {step}/{total_step}||Loss: {loss.item()} || Total Used CUDA Memory: {total_used_memory / (1024 ** 3):.2f} GB || Total CUDA Memory: {memory_total / (1024 ** 3):.2f} GB')

        if activate_wandb:
            wandb.log({"loss": loss.item(), "step": step + epoch * len(dataloader)})

    print(f'Epoch [{epoch}/{total_epochs}], Loss: {epoch_loss / len(dataloader)}')
