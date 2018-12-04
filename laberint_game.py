# -*- coding: utf-8 -*-

#author: Juan Manuel Rodríguez
import curses
from random import randint

class LaberintGame:

    levelName = 'levels/labyrinth_lvl4.txt';

    def __init__(self, board_width = 16, board_height = 6, gui = False):              
        self.score = 0
        self.done = False
        
        self.gui = gui
        self.x = 0
        self.y = 0
        self.lastAction = ''
        self.board, width, height = self.readLevel(self.levelName)
        self.board_width = {'width': width, 'height': height}        

    def readLevel(self, levelFilename):
        board = [];
        file = open(levelFilename, "r") 
        height = 0
        width = 0
        for line in file:
            board.append(list(line.rstrip()))
            height =height+1
            width = len(list(line.rstrip()))
        return board, width, height

    def start(self):
        if self.gui: self.render_init()
        self.game_init()
        return self.generate_observations()

    def game_init(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if(self.board[i][j] == 'A'):
                    self.x = i
                    self.y = j
                    return
                

    def render_init(self):
        curses.initscr()
        win = curses.newwin(self.board_width["height"] + 2, self.board_width["width"] + 2, 0, 0)
        win.nodelay(1)
        win.timeout(200)
        self.win = win
        self.render()

    def render(self):
        self.win.clear()
        self.win.border(0)
        #self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
        self.win.addstr(0, 2, 'Action: '+ self.lastAction + ' ')
        print(self.board);
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'w':
                    self.win.addch(i+1, j+1, '#')
                if self.board[i][j] != '.' and self.board[i][j] != 'w':
                    self.win.addch(i+1, j+1, self.board[i][j])
                #print(self.board[i][j])
        self.win.getch()

    def move(self, key):
        localx=-1
        localy=-1
        oldx = self.x   
        oldy = self.y

        if key == 0:
            localy = self.y-1
            self.lastAction = 'LEFT'
        elif key == 1:
            localx = self.x+1
            self.lastAction = 'DOWN'
        elif key == 2:
            localy = self.y+1
            self.lastAction = 'RIGHT'
        elif key == 3:
            localx = self.x-1
            self.lastAction = 'UP'
        
        if(localx>0 and localx<=self.board_width["width"]):
            self.x = localx
        
        if(localy>0 and localy<=self.board_width["height"]):
            self.y = localy
        
        #print("x:"+str(self.x)+" y:"+str(self.y))
        if self.board[self.x][self.y] == '.':
            self.board[oldx][oldy] = '.'
            self.board[self.x][self.y] = 'A'
        elif self.board[self.x][self.y] == 'w':
            self.x = oldx
            self.y = oldy
        elif self.board[self.x][self.y] == 't':
            self.score = self.score -100
            self.done = True
        elif self.board[self.x][self.y] == 'x':
            self.score = self.score + 100
            self.done = True            


    def step(self, key):
        # 0 - LEFT
        # 1 - DOWN
        # 2 - RIGHT
        # 3 - UP
        if self.done == True: self.end_game()
        self.move(key)        
        if self.gui: self.render()
        return self.generate_observations()


    def generate_observations(self):
        return self.done, self.score, self.board

    def render_destroy(self):
        curses.endwin()

    def end_game(self):
        if self.gui: self.render_destroy()
        raise Exception("Game over")

if __name__ == "__main__":
    game = LaberintGame(gui = True)
    game.start()    
    for _ in range(20):
       game.step(randint(0,3))
