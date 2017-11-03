#!/usr/bin/env python3
import tkinter as tk
import numpy as np
import pandas as pd
import time
import threading

from tkinter import ttk

class Value:
    def __init__(self, value=None):
        self.value = value
        self.lock = threading.Lock()

    def __call__(self, value=None):
        with self.lock:
            if value is not None:
                self.value = value
            return self.value

    def __bool__(self):
        return bool(self.value)


class Maze:
    def __init__(self, traps, goal, period=0.5):
        self.period = Value(period)  # 移动间隔时间
        self.actions = ['LEFT', 'UP', 'RIGHT', 'DOWN'] # 动作
        self.traps = traps  # 陷井
        self.goal = goal    # 目标
        self.terminals = self.traps + self.goal # 终止态

        # 创建主窗口
        window = tk.Tk()
        window.title('maze')
        window.geometry('250x380')

        w = 50  # 一个格子的宽度
        canvas = tk.Canvas(window, width=w*5, height=w*5, bd=1)
        canvas.pack()

        # 绘制格子
        grid = [[canvas.create_rectangle(j * w, i * w, j * w + w, i * w + w, fill='#EEEEEE')
                      for j in range(5)] for i in range(5)]

        # 两个陷井(2,1)和(1,3)
        canvas.create_rectangle(w, 2 * w, 2 * w, 3 * w, fill='red')
        canvas.create_line(w, 2 * w, 2 * w, 3 * w)
        canvas.create_line(w + w, 2 * w, w, 3 * w)
        canvas.create_rectangle(3 * w, w, 4 * w, 2 * w, fill='red')
        canvas.create_line(3 * w, w, 4 * w, 2 * w)
        canvas.create_line(3 * w + w, w, 3 * w, 2 * w)

        # 目标(3,3)
        canvas.create_rectangle(3 * w, 3 * w, 4 * w, 4 * w, fill='#00EE00')
        canvas.create_oval(3 * w + 5, 3 * w + 5, 4 * w - 5, 4 * w - 5, outline='#FF83FA', width=2)

        # 表示当前状态
        rec = canvas.create_rectangle(0, 0, w, w, fill='#FFFF00')

        # 显示状态的价值
        values = [[canvas.create_text(j * w + 25, i * w + 25,
                            text='0' if (i, j) not in self.terminals else '',fill='blue')
                            for j in range(5)]
                            for i in range(5)]

        # 显示q值的文本组件
        qtext = [[{a:canvas.create_text(
                    j * w if a == 'LEFT' else (j * w + w if a == 'RIGHT' else j * w + 25),
                    i * w if a == 'UP' else (i * w + w if a == 'DOWN' else i * w + 25),
                    text='0', fill='#A020F0', state=tk.HIDDEN)
                    for a in self.state_actions((i,j))}
                    for j in range(5)]
                    for i in range(5)]

        # 显示鼠标所指的状态的Q值
        canvas.bind('<Motion>', func=lambda e: self.show_q(self.position_to_state(e.x, e.y)))

        # 鼠标离开时隐藏正在显示的状态的Q值
        canvas.bind('<Leave>', func=lambda e: self.hide_q())

        # 单击左键绘制路径
        canvas.bind('<Button-1>', func=lambda e: self.draw_path(self.position_to_state(e.x, e.y)))

        # 速度调节按扭
        tk.Button(window, text='faster', command=lambda:self.change_period(0.5)).place(x=10, y=260)
        tk.Button(window, text='slower', command=lambda:self.change_period(2)).place(x=85, y=260)

        # 暂停按扭
        pause_text = tk.StringVar(value='pause')
        tk.Button(window, textvariable=pause_text, command=self.pause, width=5).place(x=160, y=260)

        # 学习方法选择
        learning_method = tk.StringVar()
        methods = ('QLearning-MC', 'TDLearning', '3STEP-TD', 'SARSA') # 下拉列表的值
        method_choosen = ttk.Combobox(window, width=12, textvariable=learning_method, values=methods, state='readonly')
        method_choosen.current(0)
        method_choosen.place(x=10, y=300)
        method_choosen.bind('<<ComboboxSelected>>', lambda e:method_choosen.selection_clear())

        # 开始按扭
        start_text = tk.StringVar(value='start')
        tk.Button(window, textvariable=start_text, command=self.start, width=5).place(x=160, y=300)

        # 显示episode和step数
        episode_text = tk.StringVar(value='episode: 0')
        tk.Label(window, textvariable=episode_text, width=12, justify=tk.LEFT).place(x=60, y=340)
        step_text = tk.StringVar(value='step: 0')
        tk.Label(window, textvariable=step_text, width=10, justify=tk.LEFT).place(x=160, y=340)

        # 消息显示框
        message_text = tk.Text(window, width=30, height=20, borderwidth=1)
        # message_text.place(x=10, y=290)

        # 点击关闭按扭事件处理
        window.protocol('WM_DELETE_WINDOW', self.close)

        self.grid = grid
        self.window = window
        self.rec = rec
        self.values = values
        self.pause_text = pause_text
        self.start_text = start_text
        self.event = threading.Event()
        self.message_text = message_text
        self.episode_text = episode_text
        self.step_text = step_text
        self.canvas = canvas
        self.w = w
        self.closed = Value(False)
        self.started = Value(False)
        self.learning_method = learning_method
        self.qtext = qtext
        self.qtext_showing = None
        self.path_lines = []
        self.path_state = None
        self.rl = None

    def position_to_state(self, x, y):
        i = int(y / self.w)
        j = int(x / self.w)
        if i > 4: i = 4
        if j > 4: j = 4
        return i,j

    def state_position(self, state):
        """
        获取某个状态的中心坐标
        """
        i,j = state
        x = j * self.w + self.w / 2
        y = i * self.w + self.w / 2
        return x,y

    def change_period(self, scale):
        self.period(self.period() * scale)

    def pause(self):
        if self.pause_text.get() == 'pause':
            self.event.clear()
            self.pause_text.set('resume')
        else:
            self.event.set()
            self.pause_text.set('pause')

    def move_to(self, state):
        i, j = state
        x1, y1 = j * self.w, i * self.w
        x2, y2 = x1 + self.w, y1 + self.w
        self.canvas.coords(self.rec, x1, y1, x2, y2)

    def print_value(self, values):
        for (i,j),v in values:
            self.canvas.itemconfig(self.values[i][j], {'text':str(round(v,2))[1:] if 0 < v < 1 else str(int(v))})
            self.canvas.itemconfig(self.grid[i][j], {'fill': Maze.color(v)})

    def print_message(self, msg):
        self.message_text.insert(index='end', chars=msg + '\n')

    def clear_message(self):
        self.message_text.delete(1.0, tk.END)

    def close(self):
        self.closed(True)
        self.event.set()
        if not self.started:
            self.quit()

    def quit(self):
        self.window.quit()

    def start(self):
        if self.start_text.get() == 'start':
            self.reset()
            self.started(True)
            self.event.set()
            method = self.learning_method.get()
            if method == 'QLearning-MC':
                self.rl = MCQLearning(self)
            elif method == 'TDLearning':
                self.rl = TDLearning(self)
            elif method == '3STEP-TD':
                self.rl = TDLearning(self, nstep=3)
            elif method == 'SARSA':
                self.rl = SARSA(self)
            self.rl.start()
            self.start_text.set('stop')
        elif self.start_text.get() == 'stop':
            self.started(False)
            self.event.set()
            self.start_text.set('start')

    def reset(self):
        for i in range(5):
            for j in range(5):
                self.canvas.itemconfig(self.values[i][j], {'text':'0' if (i,j) not in self.terminals else ''})
                self.canvas.itemconfig(self.grid[i][j], {'fill':'#EEEEEE'})
        self.move_to((0,0))
        self.pause_text.set('pause')
        self.print_step(0)
        self.print_episode(0)

    def print_episode(self, episode):
        self.episode_text.set('episode: ' + str(episode))

    def print_step(self, step):
        self.step_text.set('step: ' + str(step))

    def update_qtext(self, updated):
        """
        更新状态的Q值
        """
        for (i,j), qtable in updated:
            qtext = self.qtext[i][j]
            maxq = np.max(qtable)
            for a in qtable.index:
                q = qtable[a]
                self.canvas.itemconfig(qtext[a],
                    {'text': str(round(qtable[a],2))[1:] if 0 < q < 1 else str(int(q)),
                    'fill': '#FF00FF' if q == maxq and q > 0 else '#A020F0'})

    def show_q(self, state):
        """
        显示状态的Q值
        """
        # print('show_q:', state)
        self.hide_q()
        self.update_qtext_state(state, tk.NORMAL)
        self.qtext_showing = state

    def hide_q(self):
        """
        隐藏正在显示的状态的Q值
        """
        # print('hide_q:', self.qtext_showing)
        if self.qtext_showing is not None:
            self.update_qtext_state(self.qtext_showing, tk.HIDDEN)
            self.qtext_showing = None

    def update_qtext_state(self, state, text_state):
        i, j = state
        qtext = self.qtext[i][j]
        for text in qtext.values():
            self.canvas.itemconfig(text, {'state':text_state})

    def state_actions(self, state):
        actions = self.actions.copy()
        i,j = state
        if i == 0:
            actions.remove('UP')
        elif i == 4:
            actions.remove('DOWN')
        if j == 0:
            actions.remove('LEFT')
        elif j == 4:
            actions.remove('RIGHT')
        return actions

    def draw_path(self, state):
        if self.rl is None:
            return

        if state == self.path_state:
            self.delete_path()
            self.path_state = None
            return

        path = self.rl.best_path(state)
        # print(path)
        self.delete_path()

        for begin, end in zip(path[:-1], path[1:]):
            self.path_lines.append(self.canvas.create_line(*self.state_position(begin), *self.state_position(end), fill='#FFD39B', width=2))

        self.path_state = state

    def delete_path(self):
        for line in self.path_lines:
            self.canvas.delete(line)

    @staticmethod
    def color(value):
        c = int(255 * (1 - value) * 1.5)
        if c > 255:
            c = 255
        c = '%02x' % c
        rgb = '#' + c + 'ff' + c
        return rgb


class RL:
    def __init__(self, maze, gamma=0.9, epsilon=0.7, alpha=0.5):
        self.maze = maze
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.actions = maze.actions
        self.move_step = {'LEFT':(0,-1), 'UP':(-1,0), 'RIGHT':(0,1), 'DOWN':(1,0)}
        self.state = (0,0)
        self.traps = [(2,1), (1,3)]
        self.goal = [(3,3)]
        self.terminals = self.traps + self.goal
        self.value_star = np.zeros((5,5))
        self.states = [(i, j) for i in range(5) for j in range(5)]
        self.qtable = [[self._state_q_init((i,j)) for j in range(5)] for i in range(5)]
        self.ntable = [[self._state_q_init((i,j)) for j in range(5)] for i in range(5)]
        self.episode = 0
        self.step = 0

    def _state_q_init(self, state):
        actions = self.maze.state_actions(state)
        return pd.Series(data=np.zeros(len(actions)), index=actions)

    def move(self):
        action = self.e_greedy()
        return self.take_action(action)

    def take_action(self, action):
        next_state = self.next_state(action)
        reward = self.reward(action)
        self.move_to(next_state)
        return action, reward, next_state

    def move_to(self, next_state):
        self.state = next_state
        self.maze.move_to(next_state)

    def next_state(self, action, state=None):
        if state is None:
            state = self.state
        x,y = state
        step_x, step_y = self.move_step[action]
        return x + step_x, y + step_y

    def reward(self, action, state=None):
        next_state = self.next_state(action, state)
        if next_state in self.traps:
            return -1
        elif next_state in self.goal:
            return 1
        else:
            return 0

    def state_qtable(self, state):
        i,j = state
        return self.qtable[i][j]

    def state_ntable(self, state):
        i, j = state
        return self.ntable[i][j]

    def q(self, state, action):
        return self.state_qtable(state)[action]

    def maxq(self, state):
        return np.max(self.state_qtable(state))

    maxQ = maxq

    value = maxQ

    def update_q(self, state, action, q):
        qtable = self.state_qtable(state)
        q_eval = qtable[action]

        # q_eval = q_eval + self.alpha * (q - q_eval)

        # 取平均值
        ntable = self.state_ntable(state)
        ntable[action] += 1
        alpha = 1 / ntable[action]
        q_eval = (1 - alpha) * q_eval + alpha * q
        # q_eval = q_eval + alpha * (q - q_eval)    # 与上式等价

        self._update_q(state, action, q_eval)

    def _update_q(self, state, action, q):
        self.state_qtable(state)[action] = q


    def e_greedy(self):
        """
        使用epsilon-greedy策略选择下一步动作
        :return: 下一个动作
        """
        if np.random.random() < self.epsilon:
            # 选取Q值最大的
            return self.pi_star(self.state)
        else:
            #随机选择
            return np.random.choice(self.state_qtable(self.state).index)

    def pi_star(self, state):
        """
        使用最优策略选取动作
        """
        qtable = self.state_qtable(state)
        indexes = np.arange(len(qtable))
        np.random.shuffle(indexes)
        qtable = qtable[indexes]  # 将顺序打乱,以免值相同时总选同一个动作
        return np.argmax(qtable)

    def action_prob(self, state, action):
        """
        选择某个动作的概率值
        """
        qtable = self.state_qtable(state)
        max_action = np.argmax(qtable)
        if action == max_action:
            return self.epsilon + (1 - self.epsilon) / (len(qtable) - 1)
        else:
            return (1 - self.epsilon) / (len(qtable) - 1)

    def random_init_state(self):
        """
        随机初始化状态值
        :return:
        """
        while True:
            s = self.states[np.random.randint(25)]
            if s not in self.terminals:
                self.move_to(s)
                break

    def learning(self):
        try:
            self._learning()
            self.move_to((0,0))
        except Exception as e:
            self.maze.started(False)
            raise e
        finally:
            if self.closed():
                self.maze.quit()
        print('... Learning Ended ...')

    def _learning(self):
        pass

    def best_path(self, state, maxlen=10):
        """
        获取某个状态的最优路径，路径最长maxlen个
        """
        path = [state]
        while len(path) <= maxlen and path[-1] not in self.terminals:
            next_state = self.next_state(self.pi_star(state), state)
            if next_state in path:
                break
            path.append(next_state)
            state = next_state
        return path


    def print_updates(self, updates):
        update_values = [(state, self.maxq(state)) for state in updates]
        update_q = [(state, self.state_qtable(state)) for state in updates]
        self.maze.print_value(update_values)
        self.maze.update_qtext(update_q)

    def print_qtable(self):
        # for i,row in enumerate(self.qtable):
        #     for j,col in enumerate(row):
        #         print(str(i) + ',' + str(j) + ': ', end='')
        #         print([k + ':' + str(round(v,2)) for k,v in col.to_dict().items()])
        #     print('-' * 50)
        pass

    def update_episode(self):
        self.episode += 1
        self.maze.print_episode(self.episode)

    def update_step(self):
        self.step += 1
        self.maze.print_step(self.step)

    def wait_period(self, scale=1.0):
        if not self.closed():
            time.sleep(self.maze.period() * scale)

    def closed(self):
        return self.maze.closed

    def started(self):
        return self.maze.started and not self.closed()

    def start(self):
        threading.Thread(target=self.learning).start()


class MCQLearning(RL):
    """
    使用Monte-Carlo方法训练的QLearning
    """
    def _learning(self):
        while self.started():
            traces = self.simulate()
            if not self.started():
                break
            traces.reverse()
            updates = []
            for state, action, reward, next_state in traces:
                q = reward + self.gamma * self.maxq(next_state)
                self.update_q(state, action, q)
                updates.append(state)
                # print(state, action, q)
            self.print_updates(updates)
            self.print_qtable()
            self.update_episode()
            # self.maze.clear_message()

    def simulate(self):
        traces = []     # [(state,action,reward,next_state), ... ]
        while self.started() and self.state not in self.terminals:
            self.maze.event.wait()
            traces.append((self.state, *self.move()))
            # self.maze.print_message(', '.join(map(str, (traces[-1]))))
            self.update_step()
            self.wait_period()
        self.wait_period(0.5)
        self.random_init_state()
        return traces


class TDLearning(RL):
    """
    使用N步进行更新的TD算法
    """
    def __init__(self, maze, nstep=1):
        RL.__init__(self, maze)
        self.nstep = nstep
        self.steps = []

    def learning_nsteps(self):
        """
        使用存储的N步进行学习
        """
        q = 0
        for i, (s, a, r, s1) in enumerate(self.steps):
            q += r * (self.gamma ** i)
        q += self.maxQ(s1) * (self.gamma ** (i + 1))
        s, a, _, _ = self.steps[0]
        self.update_q(s, a, q)
        self.print_updates([s])
        self.steps = self.steps[1:]

    def learning_to_terminal(self):
        """
        走到终止态时使用存储的步骤进行学习
        """
        while len(self.steps) > 0:
            self.learning_nsteps()

    def _learning(self):
        while self.started():
            while  self.started() and self.state not in self.terminals:
                self.maze.event.wait()
                state = self.state
                action, reward, next_state = self.move()
                self.steps.append((state, action, reward, next_state))
                if len(self.steps) == self.nstep:
                    self.learning_nsteps()
                # self.maze.print_message(', '.join(map(str, (traces[-1]))))
                self.wait_period()
                self.update_step()
            self.learning_to_terminal()
            self.update_episode()
            self.wait_period(0.5)
            self.random_init_state()
            self.wait_period()


class TDLambda(RL):
    def __init__(self, maze, lambda_):
        RL.__init__(self, maze)
        self.lambda_ = lambda_
        self.steps = []

    def learning_nsteps(self):
        """
        使用存储的N步进行学习
        """
        q = 0
        for i, (s, a, r, s1) in enumerate(self.steps):
            q += r * (self.gamma ** i)
        q += self.maxQ(s1) * (self.gamma ** (i + 1))

        s, a, _, _ = self.steps[0]
        self.update_q(s, a, q)
        self.print_updates([s])

    def learning_to_terminal(self):
        """
        走到终止态时使用存储的步骤进行学习
        """
        while len(self.steps) > 0:
            self.learning_nsteps()

    def _learning(self):
        while self.started():
            while  self.started() and self.state not in self.terminals:
                self.maze.event.wait()
                state = self.state
                action, reward, next_state = self.move()
                self.steps.append((state, action, reward, next_state))
                if len(self.steps) == self.nstep:
                    self.learning_nsteps()
                # self.maze.print_message(', '.join(map(str, (traces[-1]))))
                self.update_step()
                self.wait_period()
            self.learning_to_terminal()
            self.update_episode()
            self.wait_period(0.5)
            self.random_init_state()
            self.wait_period()


class SARSA(RL):
    def __init__(self, maze):
        RL.__init__(self, maze, alpha=0.1)

    def _learning(self):
        while self.started():
            next_action = self.e_greedy()
            while  self.started() and self.state not in self.terminals:
                self.maze.event.wait()
                state = self.state
                action, reward, next_state = self.take_action(next_action)
                next_action = self.e_greedy()
                q = reward + self.gamma * self.q(next_state, next_action)
                self.update_q(state, action, q)
                self.print_updates([state])
                # self.maze.print_message(', '.join(map(str, (traces[-1]))))
                self.update_step()
                self.wait_period()
            self.update_episode()
            self.wait_period(0.5)
            self.random_init_state()
            self.wait_period()


class DQN:
    pass

if __name__ == '__main__':
    mz = Maze(traps=[(2,1), (1,3)], goal=[(3,3)])
    # rl = QLearning(maze=mz)
    # rl = TDLearning(maze=mz)
    # rl.start()
    tk.mainloop()

    print('*********** ended ***********')