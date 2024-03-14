import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Parameters
data_folder = 'caption data'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = 'caption data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation with multiple metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE_L, CIDEr
    :param beam_size: beam size at which to generate captions for evaluation
    :return: A dictionary containing all evaluation scores
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references = list()  # true captions
    hypotheses = list()  # predicted captions
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        # print("encoder_out", encoder_out.shape)
        en_out = encoder_out
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        # print("encoder_out", encoder_out.shape)
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            # print("h", h.shape)
            awe, beta = decoder.Channel_wise_attention(en_out, h)
            awe, alpha = decoder.Spatial_attention(awe, h)
            awe = awe.view(awe.shape[0], 2048, 8, 8)
            awe = decoder.AvgPool(awe)
            awe = awe.squeeze(-1).squeeze(-1)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            # print("awe", awe.shape)
            # print("gate", gate.shape)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = (top_k_words / vocab_size).long()  # (s)，将这里的除法结果转换为long类型
            next_word_inds = (top_k_words % vocab_size).long()  # (s)，确保这里的结果也是long类型

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if len(complete_seqs_scores) == 0:
            print("Warning: No complete sequences were found.")
            return 0  # 或其他合理的默认值

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

        # Convert hypotheses and references to the format expected by COCO eval tools
        refs = {}
        hyps = {}
        for i, (refs_i, hyps_i) in enumerate(zip(references, hypotheses)):
            refs[i] = [' '.join([rev_word_map[w] for w in sent]) for sent in refs_i]
            hyps[i] = ' '.join([rev_word_map[w] for w in hyps_i])

        # Calculate BLEU scores
        bleu_scores = corpus_bleu(list(refs.values()), list(hyps.values()),
                                  weights=[(1.0,), (1.0 / 2, 1.0 / 2), (1.0 / 3, 1.0 / 3, 1.0 / 3),
                                           (1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4)],
                                  smoothing_function=SmoothingFunction().method1)

        # Calculate METEOR score
        meteor_scorer = Meteor()
        # 初始化分数列表
        meteor_scores = []

        for i in range(len(hyps)):
            # 确保参考文本是字符串形式，如果是列表则转换为单个字符串
            reference_strs = [' '.join(ref) for ref in refs[i]]  # 如果refs[i]已经是字符串列表，直接使用
            hypothesis_str = hyps[i]  # 假设文本，已经是字符串形式
            score, _ = meteor_scorer.compute_score({i: reference_strs}, {i: [hypothesis_str]})
            meteor_scores.append(score)

        # 计算平均METEOR分数
        meteor_score = np.mean(meteor_scores)

        # Calculate ROUGE_L score
        rouge_scorer = Rouge()
        rouge_scores = []

        for i in range(len(hyps)):
            reference_strs = [' '.join(ref) for ref in refs[i]]  # 如果refs[i]已经是字符串列表，直接使用
            hypothesis_str = hyps[i]  # 假设文本，已经是字符串形式
            score, _ = rouge_scorer.compute_score({i: reference_strs}, {i: [hypothesis_str]})
            rouge_scores.append(score)

        # 计算平均ROUGE_L分数
        rouge_l_score = np.mean(rouge_scores)

        # Calculate CIDEr score
        cider_scorer = Cider()
        cider_scores = []

        for i in range(len(hyps)):
            reference_strs = [' '.join(ref) for ref in refs[i]]  # 如果refs[i]已经是字符串列表，直接使用
            hypothesis_str = hyps[i]  # 假设文本，已经是字符串形式
            score, _ = cider_scorer.compute_score({i: reference_strs}, {i: [hypothesis_str]})
            cider_scores.append(score)

        # 计算平均CIDEr分数
        cider_score = np.mean(cider_scores)

        # Return all scores in a dictionary
        scores = {
            'BLEU-1': bleu_scores[0],
            'BLEU-2': bleu_scores[1],
            'BLEU-3': bleu_scores[2],
            'BLEU-4': bleu_scores[3],
            'METEOR': meteor_score,
            'ROUGE_L': rouge_l_score,
            'CIDEr': cider_score
        }

    return scores


if __name__ == '__main__':
    beam_size = 2
    scores = evaluate(beam_size)
    for metric, score in scores.items():
        print(f"{metric} score @ beam size of {beam_size} is {score:.4f}.")
