from net import Alpha0_Node
from Game import Game
import pygame

cell_size = 35 #一个格子的大小
eage = 20   #边界大小
black = (0,0,0)
white = (255,255,255)
brown = (222, 184, 135)

pygame.init()
font = pygame.font.Font(None, 20)
screen = pygame.display.set_mode((cell_size*Game.size*2 + 3 * eage, cell_size*Game.size*2 + 3 * eage))
pygame.display.set_caption("AZ")

bias = [(eage,eage),(eage*2+cell_size*Game.size,eage),(eage,eage*2+cell_size*Game.size),(eage*2+cell_size*Game.size,eage*2+cell_size*Game.size)]#偏移量，右上左上右下左下
text = ["board","value_out","search_num","policy_out"]
def draw_area(sym,game:Game,node:Alpha0_Node):  
    # 画棋盘
    pygame.draw.rect(screen, brown, (bias[sym][0], bias[sym][1], cell_size*Game.size, cell_size*Game.size)) 
    for i in range(0,Game.size+1):#画横线
        pygame.draw.line(screen, black, (0+bias[sym][0], i*cell_size+bias[sym][1]), (cell_size*Game.size+bias[sym][0], i*cell_size+bias[sym][1]), 3)
    for i in range(0,Game.size+1):#画横线
        pygame.draw.line(screen, black, (i*cell_size+bias[sym][0],0+bias[sym][1]), (i*cell_size+bias[sym][0], cell_size*Game.size+bias[sym][1]), 3)

    #画标题
    text_surface = font.render(text[sym], True, black)
    screen.blit(text_surface, (bias[sym][0]+0.2*Game.size*cell_size,bias[sym][1]-eage+4))

    #画棋子
    for i in range(Game.size):
        for j in range(Game.size):
            if game.board[i][j] == 0:
                continue
            else:
                color = white if game.board[i][j] == 1 else black
                pygame.draw.circle(screen,color,(bias[sym][0] + 0.5*cell_size + i*cell_size,bias[sym][1] + 0.5*cell_size + j*cell_size),13)

    #画数值
    if sym == 0:
        return    
    for n in node.nodes:
        value=0
        i,j = n.i,n.j
        if sym == 1:
            #value = float(f"{-n.get_win_rate():.2f}")
            value = float(f"{-n.value_out:.2f}")
        elif sym == 2:
            value = n.num
        else:
            value = float(f"{node.policy_out[i*Game.size + j]:.2f}")
            
        text_surface = font.render(str(value), True, black)
        screen.blit(text_surface, (bias[sym][0] + 0.1*cell_size + i*cell_size,bias[sym][1] + 0.2*cell_size + j*cell_size))
        
def draw(game:Game,node:Alpha0_Node):
    screen.fill((255, 255, 255))
    for i in range(4):
        draw_area(i,game,node)

def wait_click():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN: 
                return event.pos
if __name__ == "__main__":
    game = Game()
    sym = 1
    end = 0
    i,j = 0,0
    node = Alpha0_Node(None,None,sym,game)
    while True:
        node.init_Nodes()
        for i in range(500):
            if i % 10 == 0:
                draw(game,node)
                pygame.display.flip()
            node.update()

        draw(game,node)
        pygame.display.flip()

        if sym == 1:   #玩家
            while True:
                pos = wait_click()
                pos = (pos[0] - eage,pos[1] - eage)
                i,j = int(pos[0]/cell_size),int(pos[1]/cell_size)
                flag,end = game.put(i,j,sym)
                if(flag):
                    break
                
        else:          #人机
            n = node.get_action(0)
            i,j = n.i,n.j

            _,end = game.put(i,j,sym)
            wait_click()
        
        draw(game,node)
        pygame.display.flip()

        sym=3-sym

        flag = True
        for n in node.nodes:
            if n.i == i and n.j == j:
                node = n
                flag = False
        if flag:
            node = Alpha0_Node(None,None,sym,game)
            
        if not end == 0:
            draw(game,node)
            pygame.display.flip()
            wait_click()
            break
