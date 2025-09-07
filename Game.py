import torch
import numpy as np

vecs = [(1,0),(0,1),(1,1),(1,-1)]

class Game():
    size = 9  # 棋盘大小
    win_num = 5
   
    def __init__(self):
        self.p_num = 0  #已经下了几个棋子
        self.board = np.zeros((self.size,self.size))

    def put(self,i:int,j:int,sym:int) -> tuple[bool,int]:#第一个值是是否放置成功，第二个是是否胜利,0未完成1当前胜利2平局
        if not Game.legal_index(i,j) or not self.board[i][j] == 0:
            return False,0
        self.board[i][j] = sym
        self.p_num += 1

        for vec in vecs:
            num = 1
            ii = i+vec[0]
            jj = j+vec[1]
            while Game.legal_index(ii,jj) and self.board[ii][jj] == sym:
                num += 1
                ii += vec[0]
                jj += vec[1]

            ii = i-vec[0]
            jj = j-vec[1]
            while Game.legal_index(ii,jj) and self.board[ii][jj] == sym:
                num += 1
                ii -= vec[0]
                jj -= vec[1]
            
            if num >= Game.win_num:
                return True,1
        
        return (True,2) if self.p_num >= Game.size**2 else (True,0)
    
    def clone(self) -> "Game":
        g = Game()
        g.p_num = self.p_num
        g.board = np.copy(self.board)
        return g
    
    def get_state(self,sym:int,flag=True) -> torch.tensor:     #二通道，自己，对手,sym是要取得状态的人的颜色,flag为是否添加批次
        out = torch.zeros(2, Game.size, Game.size,dtype=torch.float)
        for i in range(Game.size):
            for j in range(Game.size):
                p = self.board[i][j]
                if p == 0:
                   continue 
                index = 0 if p == sym else 1
                out[index][i][j] = 1
        if flag:
            out = out.unsqueeze(0)
        return out
            
    @staticmethod
    def legal_index(i:int,j:int)->bool:                #返回一个坐标是否合法
        return i < Game.size and i >= 0 and j >= 0 and j < Game.size
    
    def __repr__(self):
        out = "\n"
        for i in self.board:
            for j in i:
                out += f"{int(j)}"
            out+="\n"
        return out+"\n"
    