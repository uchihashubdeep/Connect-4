import numpy as np
import random
import matplotlib.pyplot as plt
import gzip
import shutil
import sys


class tree:
        def __init__(self,game,turn,depth,row,parent=None):
            self.game=game
            self.child=[]
            self.n=0
            self.parent=parent
            self.turn=turn   
            self.win=0
            self.draw=0
            self.loss=0
            self.reward=0
            self.depth=depth
            self.action=[]
            self.r=row

        def possible_action(self):
            self.c=[]
            for i in range(5):
                if(self.game[0][i]==0):
                    self.c.append(i)
            self.c=set_diff(self.c,self.action)
            return self.c
        def available_action(self):
            res=self.possible_action()
            return len(res)
        def get_row(self,c):
            r1=-1
            for i in range(self.r):
                if(self.game[i][c]==0):
                    r1=i
                if(self.game[i][c]!=0):
                    break
            return r1
        def make_child(self):
            if(self.available_action()==0):
                print("No expansion available in this node")
                return 
            col=self.possible_action().pop()
            self.action.append(col)
            row=self.get_row(col)
            c_game=np.copy(self.game)
            c_game[row][col]=self.turn
            if(self.turn==1):
                next_turn=2
            else:
                next_turn=1
            new_d=np.copy(self.depth)+1
            newborn=tree(c_game,next_turn,new_d,self.r,parent=self)
            self.child.append(newborn)

            return newborn
        def is_terminal(self):
            return win_check(self.game,1,self.r) or win_check(self.game,2,self.r) or full_check(self.game)
        def backward(self,winner):
            self.n=self.n+1
            if(self.turn==winner):
                self.reward=self.reward+1
                self.win=self.win+1
            if(winner==0):
                self.reward=self.reward+0.5
                self.draw=self.draw+1
            if(self.turn != winner and winner != 0):
                self.reward=self.reward-1
                self.loss=self.loss+1
            if(self.parent != None):
                self.parent.backward(winner)
        def ucti(self,c):
            ucti_weight=[]
            for i in range(len(self.child)):
                    f=((self.child[i].reward)/self.child[i].n)+(c* np.sqrt(( np.log(self.n) / (self.child[i].n))))
                    ucti_weight.append(f)
            max_w=-1000000
            max_ind=-1
            for i in range(len(ucti_weight)):
                if(ucti_weight[i]>max_w):
                    max_ind=i
                    max_w=ucti_weight[i]
            max_ar=[]
            for i in range(len(ucti_weight)):
                if(ucti_weight[i]==max_w):
                    max_ar.append(i)
            index=np.random.choice(max_ar,1)[0]
            return self.child[index]
        def ucti_0(self):
            ucti_weight=[]
            for i in range(len(self.child)):
                    f=(self.child[i].n)
                    ucti_weight.append(f)
            max_w=-1000000
            max_ind=-1
            for i in range(len(ucti_weight)):
                if(ucti_weight[i]>max_w):
                    max_ind=i
                    max_w=ucti_weight[i]
            max_ar=[]
            for i in range(len(ucti_weight)):
                if(ucti_weight[i]==max_w):
                    max_ar.append(i)
            index=np.random.choice(max_ar,1)[0]
            return self.child[index],self.action[index]

def win_check(game1,move,row):

        for c in range(2):
            for r in range(row):
                if(game1[r][c]==move and game1[r][c+1]==move and game1[r][c+2]==move and game1[r][c+3]==move):
                    return True


        if(row>=4):

            for c in range(5):
                for r in range(row-3):
                    if(game1[r][c]==move and game1[r+1][c]==move and game1[r+2][c]==move and game1[r+3][c]==move):
                        return True
            for c in range(2):
                for r in range(0,row-3):
                    if(game1[r][c]==move and game1[r+1][c+1]==move and game1[r+2][c+2]==move and game1[r+3][c+3]==move):
                        return True
            for c in range(2):
                for r in range(3,row):
                    if(game1[r][c]==move and game1[r-1][c+1]==move and game1[r-2][c+2]==move and game1[r-3][c+3]==move):
                        return True
        return False

def PrintGrid(positions):
        print('\n'.join(' '.join(str(x) for x in row) for row in positions))
        print()

def simulation(game1,move,row):
        win=False
        turn=move
        flag=0
        if(move==1):
            turn1=2
        else:
            turn1=1
        itr=0
        if(win_check(game1,turn1,row)):

            return 0,itr
        if(win_check(game1,move,row)):

            return 1,itr

        while(not(win)):
            if(turn==1):
                c=[]
                for i in range(5):
                    if(game1[0][i]==0):
                        c.append(i)
                if(len(c)==0):
                    return 2,itr
                for c2 in c:
                    r=[]
                    r1=-1
                    for i in range(row):
                        if(game1[i][c2]==0):
                            r1=i
                        if(game1[i][c2]!=0):

                            break
                    game2=np.copy(game1)
                    game2[r1][c2]=1
                    if(win_check(game2,1,row)):
                        if(move==turn):
                            flag=0
                        else:
                            flag=1
                        return flag,itr

                c1=np.random.choice(c,1)[0]

                r=[]
                r1=-1
                for i in range(row):
                    if(game1[i][c1]==0):
                        r1=i
                    if(game1[i][c1]!=0):

                        break
                game1[r1][c1]=1

                turn=2

                if(win_check(game1,1,row)):
                    win=True
                itr=itr+1
                continue

            if(turn==2):
                c=[]
                for i in range(5):
                    if(game1[0][i]==0):
                        c.append(i)
                if(len(c)==0):
                    return 2,itr
                for c2 in c:
                    r=[]
                    r1=-1
                    for i in range(row):
                        if(game1[i][c2]==0):
                            r1=i
                        if(game1[i][c2]!=0):

                            break
                        game2=np.copy(game1)
                        game2[r1][c2]=2
                        if(win_check(game2,2,row)):
                            if(move==turn):
                                flag=0
                            else:
                                flag=1
                            return flag,itr
                c1=np.random.choice(c,1)[0]
                r=[]
                r1=-1
                for i in range(row):
                    if(game1[i][c1]==0):
                        r1=i
                    if(game1[i][c1]!=0):
                        break
                game1[r1][c1]=2


                turn=1
                if(win_check(game1,2,row)):
                    win=True
                itr=itr+1

        if(win):
            if(move==turn):
                flag=0
            else:
                flag=1
        return flag,itr
def set_diff(list1,list2):
    return list(set(list1) - set(list2)) + list(set(list2) - set(list1))
def full_check(game):
        c=[]
        for i in range(5):
            if(game[0][i]==0):
                c.append(i)

        return len(c)==0
def MCTS(game,turn,plout,row):
        t=tree(game,turn,0,row)
        res=-1
        if(plout==0):
            act=t.possible_action()
            c1=np.random.choice(act,1)[0]
            gt=np.copy(game)
            r1=t.get_row(c1)
            gt[r1][c1]=turn

            t1=tree(gt,turn,0,row)
            return t1,c1
        while plout != 0:
            node=node_select(t)
            game_copy=np.copy(node.game)

            count1 = np.count_nonzero(game_copy == 1)
            count2 = np.count_nonzero(game_copy == 2)
            if(count1==count2):
                turn1=1
            else:
                turn1=2
            result,itr=simulation(game_copy,turn1,row)
            if(result==0):
                if(turn1==1):
                    node.backward(1)
                else:
                    node.backward(2)
            elif(result==2):
                node.backward(0)
            else:
                if(turn1==1):
                    node.backward(2)
                else:
                    node.backward(1)
            plout=plout-1
        return t.ucti_0()
def node_select(current):
        flag=True
        while(flag==True):
            if(len(current.possible_action()) != 0):
                child=current.make_child()
                return child
            else:
                current=current.ucti(1.414)
            if(current.is_terminal()==True):
                flag=False
        return current
class C4:
        def __init__(self,game,row):
            self.game=game
            self.row=row
        def possible_action(self,game):
            self.c=[]
            for i in range(5):
                if(game[0][i]==0):
                    self.c.append(i)
            return self.c
        def available_action(self,c):
            return len(c)
        def get_row(self,c):
            r1=-1
            for i in range(self.row):
                if(self.game[i][c]==0):
                    r1=i
                if(self.game[i][c]!=0):
                    break
            return r1
        def move(self,col,turn):
            row=self.get_row(col)
            self.game[row][col]=turn
            return self.game
def PartC():
    class QLearning:
        def __init__(self,row,plout):
            self.row=row
            self.po=plout
        def state(self,game):
            return tuple(tuple(row) for row in game)
        def possible_action(self,game):
            self.c=[]
            for i in range(5):
                if(game[0][i]==0):
                    self.c.append(i)
            return self.c
        def mirror(self,game):
            mg=np.zeros((self.row,5),dtype=int)
            for i in range(self.row):
                mg[i]=game[i][::-1]
            return mg
        def maxaQ(self,game,actionv):
            action=None
            Q_val=None
            act=self.possible_action(game)
            x=self.state(game)
            flag=0
            for a in act:
                gtemp=np.copy(game)
                gt=C4(gtemp,self.row)
                gt.move(a,2)
                gt1=np.copy(gt.game)
                x=self.state(gt1)
                if(x) in actionv:
                    if( Q_val == None or actionv[x]>=Q_val ):
                        Q_val=actionv[x]
                        action=x
            for a in act:
                gtemp=np.copy(game)
                gt=C4(gtemp,self.row)
                gt.move(a,2)
                gt1=np.copy(gt.game)
                gt2=self.mirror(gt1)
                x=self.state(gt2)
                if(x) in actionv:
                    if( Q_val == None or actionv[x]>=Q_val ):
                        Q_val=actionv[x]
                        action=x
                        flag=1
            if(flag==1):
                action=self.mirror(action)
                action=self.state(action)
            return action,Q_val,flag
        def maxQ(self,temp,actionv):
            action=None
            Q_val=None
            stat=np.asarray(temp)
            act=self.possible_action(stat)
            x=temp
            flag=0
            for a in act:
                gtemp=np.copy(stat)
                gt=C4(gtemp,self.row)
                gt.move(a,2)
                gt1=np.copy(gt.game)
                x=self.state(gt1)
                if(x) in actionv:
                    if( Q_val == None or actionv[x]>=Q_val ):
                        Q_val=actionv[x]
                        action=x
            for a in act:
                gtemp=np.copy(stat)
                gt=C4(gtemp,self.row)
                gt.move(a,2)
                gt1=np.copy(gt.game)
                gt2=self.mirror(gt1)
                x=self.state(gt2)
                if(x) in actionv:
                    if( Q_val == None or actionv[x]>=Q_val ):
                        Q_val=actionv[x]
                        action=x
                        flag=1

            if(flag==1):
                action=self.mirror(action)
                action=self.state(action)
            return action,Q_val,flag
        def episodes(self,epsilon,alpha,gamma,Q,Q2):
            reward=[0,-10,10]
            g=np.zeros(dtype='int',shape=(self.row,5))
            g1=C4(g,self.row)
            current=1
            terminate=False
            winner=0
            r=0
            itr=0
            delta=0
            sprime=None
            gt1=np.copy(g1.game)
            curr_state=self.state(gt1)
            Q[curr_state]=0
            s=curr_state
            while not terminate:
                aa=g1.possible_action(g1.game)
                posact=[]
                if(current==1):
                    mc200,act=MCTS(g1.game,1,self.po,self.row)
                    g1.game=np.copy(mc200.game)
                    gt1=np.copy(g1.game)
                    curr_state=self.state(gt1)
                    sprime=curr_state
                    if(curr_state not in Q2):
                        Q2[curr_state]=0
                else:
                    for i in range(len(aa)):
                            gtemp=np.copy(g1.game)
                            gt=C4(gtemp,self.row)
                            gt.move(aa[i],2)
                            gt1=np.copy(gt.game)
                            gt2=self.mirror(gt1)
                            mir_state=self.state(gt2)
                            curr_state=self.state(gt1)
                            if((curr_state) not in Q and (mir_state) not in Q):
                                posact.append(aa[i])
                    if(random.uniform(0,1) < epsilon):
                        act=np.random.choice(g1.possible_action(g1.game),1)[0]
                        g1.move(act,2)
                        gt1=np.copy(g1.game)
                        gt2=self.mirror(gt1)
                        mir_state=self.state(gt2)
                        curr_state=self.state(gt1)
                        sprime=curr_state
                        if(curr_state not in Q and mir_state not in Q ):
                            Q[curr_state]=0
                        if(curr_state not in Q and mir_state in Q):
                                sprime=mir_state
                    else:
                        if(len(posact) != 0):
                            at=np.random.choice(posact,1)[0]
                            g1.move(at,2)
                            gt1=np.copy(g1.game)
                            gt2=self.mirror(gt1)
                            mir_state=self.state(gt2)
                            curr_state=self.state(gt1)
                            sprime=curr_state
                            if(curr_state not in Q and mir_state not in Q):
                                Q[curr_state]=0
                            if(curr_state not in Q and mir_state in Q):
                                sprime=mir_state
                        else:
                            curr_state,Q_val1,flag=self.maxaQ(g1.game,Q)
                            sprime=curr_state

                            if(flag==1 and curr_state != None):
                                g2=np.asarray(curr_state)
                                g2=self.mirror(g2)
                                sprime=self.state(g2)
                            if(curr_state == None):
                                Q[curr_state]=0
                            gt1=np.asarray(curr_state)
                            g1.game=gt1
                #PrintGrid(g1.game)     
                win1=win_check(g1.game,1,self.row)
                win2=win_check(g1.game,2,self.row)
                draw=full_check(g1.game)
                terminate=win1 or win2 or draw
                itr=itr+1
                if(not terminate):
                    r=r-1
                if(current==1 and not terminate):
                    e1=Q[s]
                    Q[s]=Q[s]+(alpha*(-1+gamma*Q2[sprime]-Q[(s)]))
                    delta=max(delta,e1-Q[s])
                    s=sprime
                if(current==2 and not terminate):
                    Q2[s]=Q2[s]+(alpha*(-1+gamma*Q[sprime]-Q2[(s)]))
                    s=sprime
                if(current==1 and win1):
                    e1=Q[s]
                    Q2[sprime]=0
                    Q[s]=Q[s]+(alpha*(-10+gamma*(0)-Q[(s)]))
                    delta=max(delta,e1-Q[s])
                if(current==2 and win2):
                    Q[sprime]=0
                    Q2[s]=Q2[s]+(alpha*(10+gamma*(0)-Q2[(s)]))
                if(current==2 and draw and not win2):
                    Q[sprime]=0
                    Q2[s]=Q2[s]+(alpha*(0+gamma*(0)-Q2[(s)]))
                if(terminate ):
                    break
                if(current ==2):
                    current=1
                else:
                    current=2
            if(win1 ):
                return 1,r+reward[1],itr,Q,Q2,delta
            elif(win2):
                return 2,r+reward[2],itr,Q,Q2,delta
            elif(draw):
                return 0,r+reward[0],itr,Q,Q2,delta

        def train(self,alpha,gamma,epsilon,Q,Q2,lQ,y,y1,y2,win,draw,loss):
            #alpha is the learning rate,gamma is the discount factor,epsilon determines the policy
            #Q is the dictionary which saves after-states for player 2,Q2 saves afterstates for player1
            #lQ stores the length of dictionary,y stores the sum of rewards for each episode
            #y1 stores the sum of value of all states in Q,y2 stores the delta metric
            #win,draw and loss store the number of wins,losses and draws for player 2 for each batch
            pl2=0
            dr=0
            for i in range(120000):
                if(i %10000 == 0 and i != 0):
                    print(pl2,dr,10000-pl2-dr)
                    win.append(pl2)
                    draw.append(dr)
                    loss.append(10000-pl2-dr)
                    pl2=0
                    dr=0
                    alpha=alpha/np.sqrt(2)
                    print(alpha)
                    winner,r_sum,itre,Q,Q2,delta=self.episodes(epsilon,alpha,gamma,Q,Q2)
                else:
                    winner,r_sum,itre,Q,Q2,delta=self.episodes(epsilon,alpha,gamma,Q,Q2)

                if(winner==2):
                    pl2=pl2+1
                if(winner==0):
                    dr=dr+1
                if(i==119999):
                    print(pl2,dr,10000-pl2-dr)
                    win.append(pl2)
                    draw.append(dr)
                    loss.append(10000-pl2-dr)
                y.append(r_sum)
                y1.append(sum(Q.values()))
                y2.append(delta)
                lQ.append(len(Q))
                if(i<1000 or i>119000):
                    print(i,winner,len(Q),itre,delta)
                else:
                    print(i)
            return y,y1,y2,Q,Q2,lQ,win,draw,loss
        def test(self,game,Q):
            unknown=0
            s=self.state(game)
            actions=self.possible_action(game)
            max_act=-1
            max_val=-1
            max_act,max_val,flag=self.maxQ(s,Q)
            g1=C4(game,self.row)
            if(max_act==None and max_val==None):
                unknown=unknown+1
                max_act=np.random.choice(g1.possible_action(g1.game),1)
                g1.move(max_act[0],2)
            else:
                g1.game=max_act
            return g1.game,unknown
        def make_datfile(self,Q):
            with open("v4r.dat", 'w') as f:  
                for key, value in Q.items():  
                    f.write('%s:%s\n' % (key, value))
            with open('v4r.dat', 'rt') as f_in:
                with gzip.open('v4r.dat.gz', 'wt') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        def read_datfile(self,fname):
            Q_test = {}
            with gzip.open(fname,'rt') as file:
                 for line in file:
                    key, value = line.split(':')
                    #print(eval(key))
                    Q_test[eval(key)] = float(value.strip())
            return Q_test
    def Play():
        print("The Game Board has 4 rows. And maximum n for which Q learning converges is 25")
        print("If you want to play for a game with 3 rows or 2 rows, change parameter in Play()")
        i=input("How many games do you want to play? Valid inputs start from 1.")
        file="v4r.dat.gz"  #dat file name. If you want to change file, change the variable here
        Play_Game_b(int(i),4,25,file)  #first parameter is number of games, second is rows in the board, third is the opponent
        #file="v2r.dat.gz"  # dat file for two rows
        #Play_Game_b(int(i),2,25,file) # playing game for two rows
        #file="v3r.dat.gz"  # dat file for three rows
        #Play_Game_b(int(i),3,25,file) # playing game for three rows
        # the dat file for row=2,3 is shared in the report through a google drive link
    def Play_Game_b(iterations,r,opponent,fname):
        gtest4=QLearning(r,25)
        print("Starting to load values from dat file")
        Q_test=gtest4.read_datfile(fname)
        print("Values read from dat file")
        print("The size of the table is "+str(sys.getsizeof(Q_test)/(1024*1024))+" MB")
        first=0
        second=0
        draw=0
        sum_tot=0
        unknown=0
        flag=0
        winner=-1
        for k in range(iterations):
            g=np.zeros(dtype='int',shape=(r,5))
            g1=C4(g,r)
            unknown=0
            itr=0
            while(not (win_check(g1.game,1,r)or win_check(g1.game,2,r)or full_check(g1.game))):
                print('Player 1 (MCTS with '+ str(opponent)+' playouts)')
                print('Action selected : 1')
                mc200,act=MCTS(g1.game,1,opponent,r)
                playouts=mc200.n
                state1=mc200.reward/mc200.n
                g1.game=np.copy(mc200.game)
                print('Total playouts for next state: '+str(playouts))
                print('Value of next state according to MCTS : '+str(state1))
                PrintGrid(g1.game)
                if(win_check(g1.game,1,r)):
                    itr=itr+1
                    first=first+1
                    sum_tot=sum_tot+1
                    winner=1
                    break
                itr=itr+1
                print('Player 2 (Q-learning)')
                print('Action selected : 2')
                gn,flag=gtest4.test(g1.game,Q_test)
                gstate=gtest4.state(gn)
                if(flag==1):
                    unknown=unknown+1
                    print("Unknown move so value of state is unknown")
                else:
                    gstatemirror=gtest4.mirror(gstate)
                    gsmirror=gtest4.state(gstatemirror)
                    if(gstate in Q_test):
                        state2=Q_test[gstate]
                    else:
                        state2=Q_test[gsmirror]
                print('Value of next state according to Q-learning : '+str(state2))
                g1.game=gn
                PrintGrid(g1.game)
                if(win_check(g1.game,2,r)):
                    second=second+1
                    itr=itr+1
                    sum_tot=sum_tot+1
                    winner=2
                    break
                if(full_check(g1.game)):
                    itr=itr+1
                    draw=draw+1
                    sum_tot=sum_tot+1
                    winner=0
                    break
                sum_tot=sum_tot+1
                itr=itr+1

            if(winner==1):
                print('Player 1 has WON. Total moves = '+ str(itr))
                print('Unknown moves made by Player 2 = '+ str(unknown))
            elif(winner==2):
                print('Player 2 has WON. Total moves = '+ str(itr))
                print('Unknown moves made by Player 2 = '+ str(unknown))
            else:
                print('Match is DRAW. Total moves = '+ str(itr))
                print('Unknown moves made by Player 2 = '+ str(unknown))
    Play()

def PartA():
    def Play1():
        i=input("How many games do you want to play? Valid inputs start from 1.")
        Play_Game_a(int(i))   
    def Play_Game_a(g):
        first=0
        second=0
        draw=0
        winner=-1
        for k in range(g):
            g=np.zeros(dtype='int',shape=(6,5))
            g1=C4(g,6)
            itr=0
            while(not (win_check(g1.game,1,6)or win_check(g1.game,2,6)or full_check(g1.game))):
                print('Player 1 (MCTS with 200 playouts)')
                print('Action selected : 1')
                mc200,act=MCTS(g1.game,1,200,6)
                g1.game=np.copy(mc200.game)
                playouts=mc200.n
                state1=mc200.reward/mc200.n
                g1.game=np.copy(mc200.game)
                print('Total playouts for next state: '+str(playouts))
                print('Value of next state according to MCTS : '+str(state1))
                PrintGrid(g1.game)
                itr=itr+1
                if(win_check(g1.game,1,6)):
                    first=first+1
                    winner=1
                    break
                print('Player 2 (MCTS with 40 playouts)')
                print('Action selected : 2')
                mc40,act=MCTS(g1.game,2,40,6)
                g1.game=np.copy(mc40.game)
                playouts1=mc40.n
                state2=mc40.reward/mc40.n
                g1.game=np.copy(mc40.game)
                print('Total playouts for next state: '+str(playouts1))
                print('Value of next state according to MCTS : '+str(state2))
                PrintGrid(g1.game)
                itr=itr+1
                if(win_check(g1.game,2,6)):
                    second=second+1
                    winner=2
                    break
                if(full_check(g1.game)):
                    draw=draw+1
                    winner=0
                    break
                
            if(winner==1):
                print('Player 1 has WON. Total moves = '+ str(itr))
            elif(winner==2):
                print('Player 2 has WON. Total moves = '+ str(itr))
            else:c
                print('Match is DRAW. Total moves = '+ str(itr))
    Play1()

    
def main():
    flag=True
    while(flag== True):
        val=input("Enter which part you want to test. Input can be A or C.")
        if(val=="C" or val=="c"):
            flag=False
            PartC()
        elif(val=="A" or val=="a"):
            flag=False
            PartA()
        else:
            print("Invalid Input")
                

                
                
if __name__=='__main__':
    main()

