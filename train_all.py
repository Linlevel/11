import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sca_models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np
import json
import os

# Data parameters
data_folder = 'caption data/'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 32
workers = 0
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.
alpha_c = 1.
best_data = 0.
print_freq = 100
fine_tune_encoder = False
checkpoint = None


def main():
    global rev_word_map, best_data, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_data = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_metrics = validate(val_loader=val_loader,
                                  encoder=encoder,
                                  decoder=decoder,
                                  criterion=criterion)

        # Check for improvement using your own logic, possibly considering multiple metrics
        is_best = True  # You need to implement the logic to decide if the current state is the best

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_metrics, is_best)

def calculate_metrics(references, hypotheses):
    # 计算BLEU分数
    weights_for_bleu = {
        'bleu_1': (1.0, 0, 0, 0),
        'bleu_2': (0.5, 0.5, 0, 0),
        'bleu_3': (0.33, 0.33, 0.33, 0),
        'bleu_4': (0.25, 0.25, 0.25, 0.25),
    }
    bleu_scores = {key: corpus_bleu(references, hypotheses, weights=weights) for key, weights in weights_for_bleu.items()}

    # 计算METEOR分数，确保引用和假设都是字符串列表
    meteor_scores = []

    # 遍历每一对引用集和假设
    for refs, hyp in zip(references, hypotheses):
        preprocessed_refs = []
        for ref_group in refs:  # refs 应该是每个图像的引用集合，每个引用集合包含多个句子
            preprocessed_ref_group = []
            for sentence in ref_group:  # 遍历每个句子
                if isinstance(sentence, list):  # 确保sentence是一个列表
                    # 将句子中的每个词索引转换为词汇，如果索引不存在于rev_word_map中，则使用空字符串代替
                    processed_sentence = ' '.join(
                        [rev_word_map[word] if word in rev_word_map else '' for word in sentence])
                    preprocessed_ref_group.append(processed_sentence)
                else:
                    print("Warning: Expected a list of word indices, got:", type(sentence))
            preprocessed_refs.append(preprocessed_ref_group)

        # 处理假设，确保假设是词索引列表
        if isinstance(hyp, list):
            preprocessed_hyp = ' '.join([rev_word_map[word] if word in rev_word_map else '' for word in hyp])
        else:
            print("Warning: Expected a list for hypothesis, got:", type(hyp))
            continue

        # 计算单个假设的METEOR分数
        try:
            score = meteor_score(preprocessed_refs, preprocessed_hyp)
            meteor_scores.append(score)
        except ValueError as e:
            print("Error calculating METEOR score:", e)
            continue

    meteor = np.mean(meteor_scores) if meteor_scores else 0

    # 准备数据计算ROUGE和CIDEr分数
    refs = {idx: [' '.join(ref[0])] for idx, ref in enumerate(references)}
    hyps = {idx: [' '.join(hyp)] for idx, hyp in enumerate(hypotheses)}

    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(refs, hyps)

    cider = Cider()
    cider_score, _ = cider.compute_score(refs, hyps)

    return bleu_scores, meteor, rouge_score, cider_score


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = scores.data
        targets = targets.data
        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()  # 开启评估模式
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []  # 真实标注
    hypotheses = []  # 模型预测

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)
            losses.update(loss.item(), sum(decode_lengths))

            # 存储真实标注和预测
            allcaps = allcaps[sort_ind]  # 因为图像被排序了
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = []
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # 移除pads
            hypotheses.extend(temp_preds)

        # 计算所有指标
        bleu_scores, meteor, rouge_l, cider = calculate_metrics(references, hypotheses)

        print(f'\n * METEOR - {meteor:.3f}, ROUGE_L - {rouge_l:.3f}, CIDEr - {cider:.3f}')
        print(
            f' * BLEU-1 - {bleu_scores["bleu_1"]:.3f}, BLEU-2 - {bleu_scores["bleu_2"]:.3f}, BLEU-3 - {bleu_scores["bleu_3"]:.3f}, BLEU-4 - {bleu_scores["bleu_4"]:.3f}\n')

    return bleu_scores, meteor, rouge_l, cider


if __name__ == '__main__':
    main()
