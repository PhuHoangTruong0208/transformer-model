import torch
from torch import nn 
import torch.nn.functional as F
from torch.nn import Transformer
import math
import random
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# chia dữ liệu thành từng batch nhỏ khi training
"""
lớp chia batch thủ công này mình thiết kế để tối ưu việc ngốn ram của tensor, thay vì chuyển tất cả batch
sang tensor thì mình lại dùng list, sau đó trong quá trình training chia ra các batch nhỏ mình mới chuyển
các batch nhỏ đó thành tensor, điều này giúp mô hình dù training trên 1 tập dữ liệu rất lớn cũng không mất
quá nhiều ram
"""
class Dataset:
    def __init__(self, x_batch, y_batch, batch_num, random_batch=True):
        self.x_batch = x_batch
        self.y_batch = y_batch
        self.batch = []
        self.batch_num = batch_num
        self.random_batch = random_batch
        self.split_batch()

    def split_batch(self):
        for _ in range(math.ceil(len(self.x_batch)/self.batch_num)): # dùng hàm math.ceil để làm tròn số thập phân sang số gần nhất ví dụ: 7.1 -> 8
            x_batch, y_batch = [], []
            for i in range(self.batch_num):
                if self.random_batch:
                    try:
                        self.x_batch[i]
                    except:
                        break
                    x_batch.append(random.choice(self.x_batch))
                    y_batch.append(random.choice(self.y_batch))
                else:
                    try:
                        x_batch.append(self.x_batch[i])
                        y_batch.append(self.y_batch[i])
                    except:
                        break
            self.batch.append((x_batch, y_batch))

# tokenizer các ký tự trong câu
class LanguageTokenizer(nn.Module):
    def __init__(self, x_batch, y_batch, max_sequence_length, pad="<pad>", end="<end>", start="<start>", out="<out>", embedding_file="embedding.pth"):
        super().__init__()
        self.x_batch =x_batch
        self.y_batch = y_batch
        self.vocab = [pad, end, start, out]
        self.get_vocab_batch()
        self.vocab_size = len(self.vocab)
        self.text_to_number = {v:k for k,v in enumerate(self.vocab)}
        self.number_to_text = {k:v for k,v in enumerate(self.vocab)}
        self.max_sequence_length = max_sequence_length
        self.pad = pad
        self.end = end
        self.start = start
        self.out = out
        self.__sub_run__()
    
    # chạy trực tiếp các hàm cần thiết khi gọi lớp
    def __sub_run__(self):
        self.normalize_length_batch()
        self.tokenizer_batch()
        self.add_special_token()
        self.padding()
        self.normalize_2times_and_make_tensors()

    # lớp nhận từ vựng
    def get_vocab_batch(self):
        for vocab in list("".join(self.x_batch + self.y_batch)):
            if vocab not in self.vocab:
                self.vocab.append(vocab)
    
    # chuẩn hóa độ dài của câu
    def normalize_length_batch(self):
        for i in range(len(self.x_batch)):
            if len(list(self.x_batch[i])) > self.max_sequence_length:
                self.x_batch[i] = "".join(list(self.x_batch[i])[:self.max_sequence_length])
                self.y_batch[i] = "".join(list(self.y_batch[i])[:self.max_sequence_length])
    
    # chuyển các ký tự trong batch thành tokens
    def tokenizer_batch(self):
        for i in range(len(self.x_batch)):
            self.x_batch[i] = [self.text_to_number[c] for c in list(self.x_batch[i])]
            self.y_batch[i] = [self.text_to_number[c] for c in list(self.y_batch[i])]
    
    # thêm các token đặc biệt vào đầu và cuối câu của chuổi mục tiêu (Y)
    def add_special_token(self):
        for i in range(len(self.y_batch)):
            self.y_batch[i] = [2] + self.y_batch[i] + [1]
    
    # thêm đệm (số 0)
    def padding(self):
        for i in range(len(self.x_batch)):
            for _ in range(len(self.x_batch[i]), self.max_sequence_length):
                self.x_batch[i].append(0)
            for _ in range(len(self.y_batch[i]), self.max_sequence_length):
                self.y_batch[i].append(0) 
    
    # chuẩn hóa độ dài lần thứ 2 để chắc rằng mọi thứ hoạt động ổn
    def normalize_2times_and_make_tensors(self):
        for i in range(len(self.x_batch)):
            if len(self.x_batch[i]) > self.max_sequence_length:
                self.x_batch[i] = self.x_batch[i][:self.max_sequence_length]
                self.y_batch[i] = self.y_batch[i][:self.max_sequence_length]
        self.x_batch = self.x_batch
        self.y_batch = self.y_batch
    
    # tokenizer riêng cho câu đầu vào lẻ
    def tokenize_sentence(self, x):
        tokenize_sentence = []
        for token in list(x):
            try:
                tokenize_sentence.append(self.text_to_number[token])
            except:
                tokenize_sentence.append(self.text_to_number[self.out])
        tokenize_sentence = tokenize_sentence[:self.max_sequence_length]
        for _ in range(len(tokenize_sentence), self.max_sequence_length):
            tokenize_sentence.append(0)
        return torch.tensor([tokenize_sentence])

# mô hình transformer
"""
mô hình transformer mình dùng mặc định của torch, nếu muốn xem chi tiết cách xây dựng model transformer
thì mình cũng có code sẳn 1 mô hình transformer từ đầu, cứ vào đường link github này để xem và bóc tách
code 👇
https://github.com/PhuHoangTruong0208/Transformer-Finally/blob/main/transformers.py
"""
class TransformersGen(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_heads, num_enc_layers, num_dec_layers, dim_ffn, dropout, activation):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(device)
        self.transformer = Transformer(hidden_dim, num_heads, num_enc_layers, num_dec_layers, dim_ffn, dropout, activation, batch_first=True, device=device)
        self.linear_out = nn.Linear(hidden_dim, vocab_size).to(device)
    
    def forward(self, x, y, argmax=False):
        x = self.embedding(x.to(device))
        y = self.embedding(y.to(device))
        x = self.transformer(x, y)
        x = self.linear_out(x)
        x = F.log_softmax(x, dim=-1)
        if argmax:
            x = torch.argmax(x, dim=-1)
        return x

# phương thức chính chứa các chức năng chính
"""
phương thức chính, chứa tất cả các lớp đã tạo ở trên và hoạt động, có cả lớp training, lớp training này mình
thiết kế để cho nó hỏi trước khi traing, và kiểm tra model đã lưu hay chưa, bạn cũng có thể chỉnh tên file
lưu mô hình bằng tham số (model_file_name).

lưu ý mã này chỉ để học tập, nếu bạn muốn xây 1 mô hình ngôn ngữ như GPT2 hay GPT3 như của open AI
thì vẫn hoàn toàn được, mô hình này rất mạnh, bạn chỉ cần điều chỉnh các tham số như hidden dim, dim_ffn,
num_head và thêm vào một số thuật toán tìm kết quả tôt nhất của vector đầu ra thay vì argmax, ví dụ: (beam search)...vv,
thì nó hoàn toàn đủ sức để đạt đến mức độ đó, nhưng LƯU Ý Ở ĐÂY: nó cần rất rất nhiều tài nguyên
tính toán và dữ liệu để training! hãy chắc rằng bạn có 1 siêu máy tính hoặc 1 máy chủ phân tán để làm
điều đó :P
"""
class MainTransformerSystem:
    def __init__(self, x_batch, y_batch, max_sequence_length, hidden_dim, num_heads=8, num_enc_layers=6, num_dec_layers=6,
                epoch=100, learning_rate=0.001, batch_size=16, random_batch=True, verbose=True, limit_loss_break_train=1,
                model_file_name="states.pth", dim_ffn=4096, dropout=0.01, activation=F.relu):
        self.tokenizer = LanguageTokenizer(x_batch, y_batch, max_sequence_length)
        self.transformer = TransformersGen(hidden_dim, self.tokenizer.vocab_size, num_heads, num_enc_layers, num_dec_layers, dim_ffn, dropout, activation)
        self.compute_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), learning_rate)
        self.epochs = epoch
        self.batch_size = batch_size
        self.random_batch = random_batch
        self.verbose = verbose
        self.limit_loss_break_train = limit_loss_break_train
        self.model_file_name = model_file_name
        self.__sub_run__()
    
    # hỏi và chạy các task theo lệnh
    def __sub_run__(self):
        if os.path.exists(self.model_file_name):
            inp = input("đã có model đã lưu, bạn có muốn dùng (muốn/không) : ").lower().strip()
            if inp in "muốn":
                self.transformer.load_state_dict(torch.load(self.model_file_name))
        inp = input("bạn có muốn training? (muốn/không) : ").lower().strip()
        if inp in "muốn":
            self.training()
    
    # lớp đào tạo (training)
    def training(self):
        for epoch in range(self.epochs):
            dataset = Dataset(self.tokenizer.x_batch, self.tokenizer.y_batch, self.batch_size, random_batch=True)
            for i, (x_batch, y_batch) in enumerate(dataset.batch):
                # chuyển các batch nhỏ (dạng list) thành tensor
                x_batch = torch.tensor(x_batch)
                y_batch = torch.tensor(y_batch)
                self.transformer.train()
                self.optimizer.zero_grad()
                predict = self.transformer(x_batch, y_batch)
                loss = self.compute_loss(predict.view(-1, self.tokenizer.vocab_size).to(device), y_batch.view(-1).to(device))
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.loss = loss.item()
                if self.verbose:
                    print(f"Batch {i} Loss {loss.item()}")
            if self.verbose:
                print(f"complete {epoch}/{self.epochs}")
            if self.loss < self.limit_loss_break_train:
                if self.verbose:
                    self.transformer.to(torch.device('cpu'))
                    torch.save(self.transformer.state_dict(), self.model_file_name)
                    self.transformer.to(device)
                    print("mất mát đã giảm đúng như số mong muốn nên sẽ dừng traning lập tức")
                break
            self.transformer.to(torch.device('cpu'))
            torch.save(self.transformer.state_dict(), self.model_file_name)
            self.transformer.to(device)
    
    # chuyển hóa các tokens của đầu ra dự đoán sang text
    def decode_to_language(self, x):
        sentence = ""
        for token in x[0]:
            sentence += self.tokenizer.number_to_text[int(token)]
        return sentence
    
    # quá trình hoạt động, dự đoán, chatting
    def chatting_predict(self, x):
        y = self.tokenizer.tokenize_sentence("<start>")
        x = self.tokenizer.tokenize_sentence(x)
        x = self.transformer(x, y, argmax=True)
        out = self.decode_to_language(x)
        return out

# mẫu
# x, y = ["hello", "what is your name"], ["hi", "my name is h"]
# model = MainTransformerSystem(x, y, max_sequence_length=300, hidden_dim=32, num_heads=4, num_dec_layers=2,
#                         num_enc_layers=2, dim_ffn=64, epoch=15, dropout=0.01, learning_rate=0.0001, limit_loss_break_train=0)
# result = model.chatting_predict("hello")
# print(result)