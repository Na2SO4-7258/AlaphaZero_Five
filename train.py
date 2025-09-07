from Game import Game
from net import Alpha0_Node
import net as net
import json
import numpy as np

def one_game(show:bool):
    game = Game()
    game.put(4,4,2)

    buffer = [[],[]]
    sym = 1
    end = False
    step = 0

    while not end:
        step+=1
        tree = Alpha0_Node(None,None,sym,game)
        tree.init_Nodes()
        for i in range(300):
            tree.update()

        search_rate = np.zeros((Game.size,Game.size))
        for node in tree.nodes:
            search_rate[node.i][node.j] = node.num/tree.num
            

        buffer[sym - 1].append({
            "state" : game.get_state(tree.sym).tolist(),
            "search_rate" : search_rate.tolist(),
            "win_rate" : tree.get_win_rate()
        })       


        t = 0 if step > 5 else 1
        #t=0

        node = tree.get_action(t)
        i,j = node.i,node.j

        _,end = game.put(i,j,tree.sym)
            
        sym = 3-sym
    #加入终局
    buffer[sym - 1].append({
        "state" : game.get_state(sym).tolist(),
        "search_rate" : np.zeros((Game.size,Game.size)).tolist(),
        "win_rate" : -1 if end == 1 else 0#对手赢了
    })

    if end == 1:
        win = 3-sym#如果在1的回合赢了sym最后会变成2，所以反转回来
        print(f"{win}赢了")
        reward = 1
        for i in range(len(buffer[win-1])-1,-1,-1):
            buffer[win-1][i]["win_rate"] += reward
            reward *= 0.6
            reward = max(0.2,reward)
            buffer[win-1][i]["win_rate"] = min(1,buffer[win-1][i]["win_rate"])

        reward = 1
        for i in range(len(buffer[sym-1])-1,-1,-1):
            buffer[sym-1][i]["win_rate"] -= reward
            reward *= 0.6
            reward = max(0.2,reward)
            buffer[sym-1][i]["win_rate"] = max(-1,buffer[sym-1][i]["win_rate"])


    for i in buffer:
        for j in i:
            net.model.add_buffer(j)
            

if __name__ == "__main__":
    for i in range(100000):
        print(f"第{i}局")
        show = True if i%10 == 0 else False
        one_game(show)
        if i >= 10 and i % 10 == 0:
            net.model.train_model(2000)
        if i%300 == 0  and i > 0:
            with open("data/d1.json", "w", encoding="utf-8") as f:
                json.dump(net.model.buffers, f, ensure_ascii=False, indent=4)