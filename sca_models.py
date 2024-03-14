import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    shift to only output the feature map
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet152(pretrained=True)  # 预训练ImageNet ResNet-101

        # 删除线性层和池层(因为我们不做分类)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)# 用于按顺序封装一系列子模块（例如层、激活函数等）

        # 将图像大小调整为固定大小以允许输入可变大小的图像
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: 图像，一个维度张量(batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        feature_map = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        #out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return feature_map

    def fine_tune(self, fine_tune=False):
        """
        阻止编码器的卷积块2到4的梯度计算。

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # 如果微调，只微调卷积块2到4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Spatial_attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self,feature_map,decoder_dim,K = 512):
        """
        :param feature_map: feature map in level L
        :param decoder_dim: size of decoder's RNN
        """
        super(Spatial_attention, self).__init__()
        _,C,H,W = tuple([int(x) for x in feature_map])#这行代码从feature_map参数中提取通道数C、高度H和宽度W，并将它们转换为整数
        self.W_s = nn.Parameter(torch.randn(C,K))#权重参数
        self.W_hs = nn.Parameter(torch.randn(K,decoder_dim))
        self.W_i = nn.Parameter(torch.randn(K,1))
        self.bs = nn.Parameter(torch.randn(K))#偏置参数
        self.bi = nn.Parameter(torch.randn(1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)  # Softmax层来计算权重

    def forward(self, feature_map, decoder_hidden):
        """
        Forward propagation.

        :param feature_map: L层的特征映射(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(feature_map.shape[0],2048,-1)# 将特征映射的形状调整为(batch_size, 2048, W*H)
        # print("V_map shape:", V_map.shape)#([8, 2048, 64])
        V_map = V_map.permute(0,2,1)#通过改变维度的顺序，将V_map的形状调整为(batch_size, W*H, C)
        # print("V_map1 shape:", V_map.shape)#([8, 64, 2048])
        # print("m1",torch.matmul(V_map,self.W_s).shape)
        # print("m2",torch.matmul(decoder_hidden,self.W_hs).shape)
        att = self.tanh((torch.matmul(V_map,self.W_s)+self.bs) + (torch.matmul(decoder_hidden,self.W_hs).unsqueeze(1)))#(batch_size,W*H,C)
        # 第一部分是将变换后的特征映射V_map与权重W_s相乘并加上偏置bs。第二部分是将解码器隐藏状态decoder_hidden与权重W_hs相乘，结果增加一个维度后相加。
        # print("att",att.shape)#([8, 64, 512])
        alpha = self.softmax(torch.matmul(att,self.W_i) + self.bi)
        # print("alpha",alpha.shape)#([8, 64, 1])
        alpha = alpha.squeeze(2)# 移除alpha中长度为1的维度，使其维度与feature_map相匹配
        feature_map = feature_map.view(feature_map.shape[0],2048,-1)
        # print("feature_map",feature_map.shape)#([8, 2048, 64])
        # print("alpha",alpha.shape)#([8, 64])
        temp_alpha = alpha.unsqueeze(1)#给alpha增加一个维度，以便能够与feature_map进行逐元素乘法。
        # print("temp_alpha", temp_alpha.shape)
        attention_weighted_encoding = torch.mul(feature_map,temp_alpha)
        return attention_weighted_encoding,alpha


class Channel_wise_attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self,feature_map,decoder_dim,K = 512):
        """
        :param feature_map: feature map in level L
        :param decoder_dim: size of decoder's RNN
        """
        super(Channel_wise_attention, self).__init__()
        _,C,H,W = tuple([int(x) for x in feature_map])
        self.W_c = nn.Parameter(torch.randn(1,K))#权重参数（区别）
        self.W_hc = nn.Parameter(torch.randn(K,decoder_dim))
        self.W_i_hat = nn.Parameter(torch.randn(K,1))
        self.bc = nn.Parameter(torch.randn(K))#偏置参数
        self.bi_hat = nn.Parameter(torch.randn(1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)  # softmax layer to calculate weights

    def forward(self, feature_map, decoder_hidden):
        """
        Forward propagation.

        :param feature_map: feature map in level L(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(feature_map.shape[0],2048,-1) .mean(dim=2)
        V_map = V_map.unsqueeze(2)#(batch_size,C,1)区别
        # print(feature_map.shape)
        # print(V_map.shape)#([8, 2048, 1])
        # print("wc",self.W_c.shape)
        # print("whc",self.W_hc.shape)
        # print("decoder_hidden",decoder_hidden.shape)
        # print("m1",torch.matmul(V_map,self.W_c).shape)
        # print("m2",torch.matmul(decoder_hidden,self.W_hc).shape)
        # print("bc",self.bc.shape)
        att = self.tanh((torch.matmul(V_map,self.W_c) + self.bc) + (torch.matmul(decoder_hidden,self.W_hc).unsqueeze(1)))#(batch_size,C,K)
        # print("att",att.shape)
        beta = self.softmax(torch.matmul(att,self.W_i_hat) + self.bi_hat)
        # print("beta", beta.shape)
        beta = beta.unsqueeze(2)
        # print("beta1",beta.shape)
        attention_weighted_encoding = torch.mul(feature_map,beta)

        return attention_weighted_encoding,beta


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    shift to sca attention
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,encoder_out_shape=[1,2048,8,8], K=512,encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout


        self.Spatial_attention = Spatial_attention(encoder_out_shape, decoder_dim, K)  # attention network
        self.Channel_wise_attention = Channel_wise_attention(encoder_out_shape, decoder_dim, K) # ATTENTION
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, 1000, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.AvgPool = nn.AvgPool2d(8)
    def init_weights(self):
        """
       用均匀分布的值初始化一些参数，以便于收敛。
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_emmbeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        基于编码图像为解码器的LSTM创建初始隐藏状态和单元状态。

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = self.AvgPool(encoder_out).squeeze(-1).squeeze(-1)
        # print("mean_encoder_out shape:", mean_encoder_out.shape)#([8, 2048])
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # print("caption_lengths:", caption_lengths)#([14, 14, 14, 14, 14, 14, 14, 14])
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # print("encoded_captions shape:", encoded_captions.shape)#([8, 52])
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # print("embeddings shape:", embeddings.shape)# ([8, 52, 512])
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        # print("h shape:", h.shape)#([8, 512])
        # print("c shape:", c.shape)#([8, 512])
        # 我们不会在<end>位置进行解码，因为一旦生成<end>，我们就完成了生成。
        # 因此，解码长度是实际长度- 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to 保存单词预测分数和alpha值
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)#需要更改形状？
        #alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)#需要更改形状

        # At each time-step, decode by
        # attention-weighing 基于解码器先前隐藏状态输出的编码器输出
        # 然后在解码器中使用前一个单词和注意加权编码生成一个新词
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
            #                                                     h[:batch_size_t])
            #channel-spatial模式attention
            #channel_wise
            attention_weighted_encoding, beta = self.Channel_wise_attention(encoder_out[:batch_size_t],h[:batch_size_t])
            # print("C_att_weighted shape:", attention_weighted_encoding.shape)#([8, 2048, 8, 8])
            #spatial
            attention_weighted_encoding, alpha = self.Spatial_attention(attention_weighted_encoding[:batch_size_t],h[:batch_size_t])
            # print("S_att_weighted shape:", attention_weighted_encoding.shape)#([8, 2048, 64])
            #对attention_weighted_encoding降维
            attention_weighted_encoding = attention_weighted_encoding.view(attention_weighted_encoding.shape[0],2048,8,8)
            # channel_wise
            attention_weighted_encoding, beta = self.Channel_wise_attention(attention_weighted_encoding[:batch_size_t],
                                                                            h[:batch_size_t])
            # print("C_att_weighted shape:", attention_weighted_encoding.shape)#([8, 2048, 8, 8])
            # spatial
            attention_weighted_encoding, alpha = self.Spatial_attention(attention_weighted_encoding[:batch_size_t],
                                                                        h[:batch_size_t])
            # 对attention_weighted_encoding降维
            attention_weighted_encoding = attention_weighted_encoding.view(attention_weighted_encoding.shape[0], 2048,  8, 8)
            # print("att_weighted1 shape:", attention_weighted_encoding.shape)#([8, 2048, 8, 8])
            attention_weighted_encoding = self.AvgPool(attention_weighted_encoding)
            # print("att_weighted2 shape:", attention_weighted_encoding.shape)#([8, 2048, 1, 1])
            attention_weighted_encoding = attention_weighted_encoding.squeeze(-1).squeeze(-1)
            # print("att_weighted3 shape:", attention_weighted_encoding.shape)#([8, 2048])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            #alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, sort_ind

# #--------------------------------Encoder----------------------------------------
# encoder = Encoder(encoded_image_size=14)  # 这里14是示例值，可以根据需要调整
# # 生成输入数据
# batch_size = 8  # 示例批量大小
# input_images = torch.rand(batch_size, 3, 256, 256)  # 假设输入图像尺寸为 256x256
# # 执行前向传播
# output_features = encoder(input_images)
# # 输出结果的维度
# # print(f"Input dimensions: {input_images.shape}")#([8, 3, 256, 256])
# # print(f"Output dimensions: {output_features.shape}")#([8, 2048, 8, 8])
# #----------------------Spatial_att----------------------------------------
# # image_size = 256
# # decoder_dim = 512
# # # 初始化模拟输入
# # images = torch.rand(batch_size, 3, image_size, image_size)  # (8, 3, 256, 256)
# # decoder_hidden = torch.rand(batch_size, decoder_dim)  # (8, 512)
# # # Encoder 实例化和前向传播
# # encoder = Encoder(encoded_image_size=14)
# # encoder_output = encoder(images)
# # # 输出Encoder输出维度
# # # print("Encoder output shape:", encoder_output.shape)#([8, 2048, 8, 8])
# # # 根据Encoder输出计算Spatial_attention所需feature_map的模拟值
# # _, C, H, W = encoder_output.shape
# # feature_map = (batch_size, C, H, W)
# # # Spatial_attention 实例化和前向传播
# # spatial_attention = Spatial_attention(feature_map, decoder_dim)
# # attention_weighted_encoding, alpha = spatial_attention(encoder_output, decoder_hidden)
# # # 输出Spatial_attention输出维度
# # # print("Attention weighted encoding shape:", attention_weighted_encoding.shape)#([8, 2048, 64])
# # # print("Alpha shape:", alpha.shape)#([8, 64])
# # #----------------------Channel_wise_att----------------------------------------
# # image_size = 256
# # decoder_dim = 512
# # # 初始化模拟输入
# # images = torch.rand(batch_size, 3, image_size, image_size)  # (8, 3, 256, 256)
# # decoder_hidden = torch.rand(batch_size, decoder_dim)  # (8, 512)
# # # Encoder 实例化和前向传播
# # encoder = Encoder(encoded_image_size=14)
# # encoder_output = encoder(images)
# # # 输出Encoder输出维度
# # # print("Encoder output shape:", encoder_output.shape)#([10, 2048, 8, 8])
# # # 根据Encoder输出计算Spatial_attention所需feature_map的模拟值
# # _, C, H, W = encoder_output.shape
# # feature_map = (batch_size, C, H, W)
# # # Channel_wise_attention
# # channel_wise_attention = Channel_wise_attention(feature_map, decoder_dim)
# # attention_weighted, beta = channel_wise_attention(encoder_output, decoder_hidden)
# # # 输出Spatial_attention输出维度
# # print("Attention weighted encoding shape:", attention_weighted.shape)#([8, 2048, 8, 8])
# # print("Bate shape:", beta.shape)#([8, 2048, 1, 1])
# #------------------------DecoderWithAttention------------------------------------------
# import torchvision.transforms as transforms
# from datasets import *
# # Data parameters
# data_folder = 'caption data/'  # folder with data files saved by create_input_files.py
# data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# batch_size = 8
# max_caption_length = 52
# word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
# with open(word_map_file, 'r') as j:
#     word_map = json.load(j)
# # print(len(word_map))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# transform = transforms.Compose([normalize])
# # 根据实际情况更新数据集初始化
# dataset = CaptionDataset(
#     data_folder,  # 数据存储位置
#     data_name,    # 数据集名称
#     'TRAIN',      # 使用训练集
#     transform=transform
# )
#
# index = 0
# img, cap, caplen = dataset[index]
# # # 输出样本维度
# # print("Image shape:", img.shape)
# # print("Caption shape:", cap.shape)
# # print("Caption length:", caplen)
#
# encoded_captions = torch.randint(0, len(word_map), (batch_size, max_caption_length))#([8, 52])
# caption_lengths = torch.LongTensor([14] * batch_size).view(batch_size, 1)#[[14],[14],[14],[14],[14],[14],[14],[14]]
# encoder_out = torch.rand(batch_size, 2048, 8, 8)  # 假设的Encoder输出
# # print("Img shape:", encoded_captions.shape)
# # print("Cap shape:", encoder_out.shape)
# # print("Caplen:", caption_lengths)
# # 初始化DecoderWithAttention
# decoder = DecoderWithAttention(
#     attention_dim=512,
#     embed_dim=512,
#     decoder_dim=512,
#     vocab_size=len(word_map),#2633
#     dropout=0.5
# )
# # 前向传播
# predictions, encoded_captions_sorted, decode_lengths, sort_ind = decoder(encoder_out, encoded_captions, caption_lengths)
# # 输出维度
# # print("Predictions shape:", predictions.shape)#([8, 13, 2633])
# # print("Encoded captions sorted shape:", encoded_captions_sorted.shape)#([8, 52])
# # print("Decode lengths:", decode_lengths)#[13, 13, 13, 13, 13, 13, 13, 13]
# # print("Sort indices:", sort_ind)#([8])
