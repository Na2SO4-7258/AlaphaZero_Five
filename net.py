import torch
from Game import Game
from MCTS.Tree import Node
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

use_noise = True

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(x))
        out += identity   
        return F.relu(out)
    
class Alpha0_Module(torch.nn.Module):
    def __init__(self):
        super(Alpha0_Module, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # backbone
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(128)

        # policy head: 输出 [B, H, W]
        self.policy_conv = nn.Conv2d(128, 1, kernel_size=1)

        # value head: 先池化成全局，再MLP
        self.value_fc1  = nn.Linear(128, 128)
        self.value_fc2  = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), 0.001)


        try:
            self.load_state_dict(torch.load("model.pth"))
            print("成功加载模型")
        except Exception as e:
           print(f"未加载数据,原因:{e}")

        self.buffers = []
        self.buffersize = 50000
        self.bufferpointer = 0

        self.to(self.device) 

        try:
            for i in range(1, 2):
                path = f"data/d{i}.json"
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.buffers.extend(data)   

            self.bufferpointer = len(self.buffers)
            if len(self.buffers) >= self.buffersize:
                self.buffers = self.buffers[len(self.buffers) - self.buffersize : self.buffersize]
                self.bufferpointer = 0

            print(f"已加载{len(self.buffers)}条数据")
        except Exception as e:
           print(f"未加载数据,原因:{e}")

   
    def forward(self, x):
        x = x.to(self.device)
        self.to(self.device)

        # backbone
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.relu(self.conv2(x))
        x = self.res2(x)

        # policy head
        p = self.policy_conv(x)              
        policy_logits = p.view(p.size(0), -1)  

        # value head
        v = F.adaptive_avg_pool2d(x, (1, 1))   
        v = v.view(v.size(0), -1)              
        v = self.relu(self.value_fc1(v))       
        value = torch.tanh(self.value_fc2(v))  

        return policy_logits, value

    def train_model(self, epochs: int):
        self.to(self.device)   # 确保模型在 self.device
        self.train()

        for epoch in range(epochs):
            batch = random.sample(self.buffers, min(128, len(self.buffers)))
            states,target_policies,target_values = [],[],[]

            n = int(random.random()*3)
            n1 = random.random()
            for data in batch:
                state = torch.tensor(data["state"],dtype=torch.float32).view(2, Game.size, Game.size)  #[2,size,size]
                policy = torch.tensor(data["search_rate"],dtype=torch.float32)                         #[size,size]

                #数据增强
                state = torch.rot90(state, k=n, dims=(1, 2))  
                policy = torch.rot90(policy, k=n, dims=(0, 1)) 
                if n1 < 0.5:  
                    state = torch.flip(state, dims=[2])  
                    policy = torch.flip(policy, dims=[1]) 


                states.append(state)
                target_policies.append(policy.reshape(-1))
                target_values.append(torch.tensor(data["win_rate"],dtype=torch.float32))

            states = torch.stack(states).to(self.device)
            target_policies = torch.stack(target_policies).to(self.device)
            target_values = torch.stack(target_values).unsqueeze(-1).to(self.device)
          

            # forward
            policy_logits, values = self(states)  # forward 已保证输入和模型同设备

            policy_loss = -(target_policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

            value_loss = F.mse_loss(values, target_values)

            loss = policy_loss + 0.4*value_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 200 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, "
                    f"Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}")

        self.to("cpu")
        torch.save(self.state_dict(), "model.pth")
        print("模型已保存")
        torch.cuda.empty_cache()
        self.eval()


    def add_buffer(self,data):
        if len(self.buffers) < self.buffersize:
            self.buffers.append(data)
        else:
            self.buffers[self.bufferpointer] = data
            self.bufferpointer = (self.bufferpointer+1)%self.buffersize

model = Alpha0_Module() 

class Alpha0_Node(Node):

    def __init__(self,i:int,j:int,sym:int,game:Game):
        super().__init__()
        self.i, self.j, self.sym = i, j, sym
        self.game : Game = game.clone()
        self.end = 0#1输,2平,3赢
        self.model = model
        # 如果不是根节点，先把落子执行到棋盘
        if self.i is not None:
            _,self.end = self.game.put(self.i, self.j, 3-self.sym)
        
        if self.fnode == None: #祖宗节点只能单独计算
            with torch.no_grad():
                state = self.game.get_state(self.sym).to(self.model.device)
                self.policy_out, self.value_out = self.model(state)
                self.policy_out = self.policy_out[0]
                self.value_out = self.value_out.item() 
                for i in range(Game.size):
                    for j in range(Game.size):
                        if not self.game.board[i][j] == 0:
                            self.policy_out[i*Game.size+j] = -1e9
                self.policy_out = F.softmax(self.policy_out,dim=-1).cpu().numpy()



    def init_Nodes(self) -> int:
        if not self.end == 0:
            return 0
        if not len(self.nodes) == 0:
            return len(self.nodes)
        
        states = []
        for i in range(Game.size):
            for j in range(Game.size):
                if self.game.board[i][j] == 0:
                    node = Alpha0_Node(i,j,3-self.sym,self.game)
                    node.fnode = self
                    self.nodes.append(node)
                    self.unsearch_nodes.append(node)

                    if node.end == 1:                   #如果子节点赢了，证明这步棋赢了，没必要再探索
                        node.value_out = -1
                        self.end =3
                        self.value_out = 1
                        return 0

                    states.append(node.game.get_state(node.sym,False))
        
        with torch.no_grad():
            states = torch.stack(states).float().to(self.model.device)
            policy,value = self.model.forward(states)

            masks = []
            for node in self.nodes:
                mask = torch.from_numpy(node.game.board.flatten() != 0)
                masks.append(mask)
            masks = torch.stack(masks).to(dtype=torch.bool, device=model.device) 


            policy[masks] = -1e9
            policy = F.softmax(policy, dim=-1)
            policy = policy.cpu().numpy()
            value = value.cpu().squeeze(-1).numpy()

        for i in range(len(self.nodes)):
            self.nodes[i].policy_out = policy[i]
            self.nodes[i].value_out = float(value[i])
            self.nodes[i].cal_model_out()

        return len(self.nodes)

    def cal_model_out(self):                   
        if self.end == 1:                       #如果赢了肯定是上一次落子，也就是对于这个节点来说对方的颜色赢了，所以反馈为-1
            self.value_out = -1
        elif self.end == 2:
            self.value_out = 0
        
    
    def random_play(self) -> float:#对于父节点
        return self.value_out
 
    def get_node(self):
        win_rates = np.array([-n.get_win_rate() for n in self.nodes], dtype=np.float32)
        nums = np.array([n.num for n in self.nodes], dtype=np.float32)
        idxs = np.array([n.i * Game.size + n.j for n in self.nodes], dtype=np.int32)
        priors = self.policy_out[idxs] 

        uct = win_rates + self.c * priors * math.sqrt(self.num) / (1.0 + nums)
        best_idx = np.argmax(uct)
        return self.nodes[best_idx]
    
    def get_action(self, tau: float = 1):
        counts = np.array([node.num for node in self.nodes], dtype=np.float32)

        if tau == 0:
            # 贪心：选择访问次数最多的
            idx = np.argmax(counts)
        else:
            # 温度采样
            probs = counts ** (1.0 / tau)
            probs /= probs.sum()
            idx = np.random.choice(len(self.nodes), p=probs)

        return self.nodes[idx]
    
    def __repr__(self):
        out = "\n"

        out += "子节点模型胜率:\n"
        win_rate = [[0.00 for _ in range(Game.size)] for _ in range(Game.size)]
        for node in self.nodes:
            win_rate[node.i][node.j] = -node.value_out
        for vec in win_rate:
            for i in vec:
                out += f"{i:.2f}  "
            out += "\n"  

        out += "子节点探索胜率:\n"
        win_rate = [[0.00 for _ in range(Game.size)] for _ in range(Game.size)]
        for node in self.nodes:
            win_rate[node.i][node.j] = -node.get_win_rate()
        for vec in win_rate:
            for i in vec:
                out += f"{i:.2f}  "
            out += "\n"  

        out += "子节点探索次数:\n"
        nums = [[0 for _ in range(Game.size)] for _ in range(Game.size)]
        for node in self.nodes:
            nums[node.i][node.j] = node.num
        for vec in nums:
            for i in vec:
                out += f"{i}  "
            out += "\n"  



        out += "\n"
        return out
    
if __name__ == "__main__":
    model.train_model(5000) 
