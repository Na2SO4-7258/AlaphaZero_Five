from __future__ import annotations
import math

class Node():
    def __init__(self):
        self.num : int = 0            #访问次数 
        self.nodes : list[Node] = []  #子节点
        self.unsearch_nodes : list[Node] = [] #还未探索过的子节点
        self.reward = 0             #收益
        self.c = 0.3                  #探索率
        self.fnode:Node = None        #父节点

    def init_Nodes(self) -> int:    #返回节点个数
        return 0


    def update(self) -> None:
        if not len(self.unsearch_nodes) == 0:
            node = self.unsearch_nodes.pop()  
            reward = node.random_play()
            node.backward(reward)
        else:
            node:Node = self.get_node()
            if(node.init_Nodes() == 0):
                reward = node.random_play()
                node.backward(reward)
            else:
                node.update()


    def get_win_rate(self) -> int:#返回胜率
        if self.num == 0:
            return 0
        return self.reward/self.num

    def random_play(self) -> float:#返回当前节点的收益
        return 1    
    
    def get_node(self) -> Node:  #返回要探索的节点
        max,out = -999999,None
        for node in self.nodes:
            win_rate = -node.get_win_rate()
            value = win_rate + self.c * math.sqrt(math.log(self.num + 1) / (node.num + 1))
            if value > max:
                max = value
                out = node
        return out
    
    def backward(self,reward:float) -> None:
        node = self
        while node is not None:
            node.num += 1
            node.reward += reward
            reward *= -1
            node = node.fnode

    def get_action(self) -> Node:  #返回胜率最高的节点
        max,out = -1,None
        for node in self.nodes:
            if node.num >= max:
                max = node.num
                out = node
        return out


                
