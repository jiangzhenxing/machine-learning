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
    def __init__(self, traps, goal, period=0.5, color_scale=1):
        self.period = Value(period)  # 移动间隔时间
        self.actions = ['LEFT', 'UP', 'RIGHT', 'DOWN'] # 动作
        self.traps = traps  # 陷井
        self.goal = goal    # 目标
        self.terminals = self.traps + self.goal # 终止态
        self.color_scale = color_scale

        # 创建主窗口
        window = tk.Tk()
        window.title('maze')
        window.geometry('250x370')

        w = 50  # 一个格子的宽度
        canvas = tk.Canvas(window, width=w*5, height=w*5, bd=1)
        canvas.pack()

        # 绘制格子
        grid = [[canvas.create_rectangle(j * w, i * w, j * w + w, i * w + w, fill='#EEEEEE') for j in range(5)] for i in range(5)]

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
        rec = canvas.create_rectangle(0, 0, w, w, fill='#FFFF00', state=tk.HIDDEN)

        # 显示状态的价值
        # 5x5列表，坐标与状态相对应，每个元素为显示价值的文本，终止态的价值不显示
        values = [[canvas.create_text(j * w + 25, i * w + 25, text='0' if (i, j) not in self.terminals else '',fill='blue') for j in range(5)] for i in range(5)]

        # 显示q值的文本组件
        # 5x5列表，坐标与状态相对应，每个元素为action:text的dict，并根据action对文本的位置进行调整
        qtext = [[{a: canvas.create_text(j * w if a == 'LEFT' else (j * w + w if a == 'RIGHT' else j * w + 25), i * w if a == 'UP' else (i * w + w if a == 'DOWN' else i * w + 25), text='', fill='#A020F0', state=tk.HIDDEN) for a in self.state_actions((i,j))} for j in range(5)] for i in range(5)]

        # 显示鼠标所指的状态的Q值
        canvas.bind('<Motion>', func=lambda e: self.show_q(self.position_to_state(e.x, e.y)))

        # 鼠标离开时隐藏正在显示的状态的Q值
        canvas.bind('<Leave>', func=lambda e: self.hide_q())

        # 单击左键绘制路径，再次单击清除路径
        canvas.bind('<Button-1>', func=lambda e: self.draw_path(self.position_to_state(e.x, e.y)))

        # 速度调节按扭
        tk.Button(window, text='faster', command=lambda:self.change_period(0.5)).place(x=10, y=260)
        tk.Button(window, text='slower', command=lambda:self.change_period(2)).place(x=85, y=260)

        # 暂停按扭
        pause_text = tk.StringVar(value='pause')
        tk.Button(window, textvariable=pause_text, command=self.pause, width=5).place(x=160, y=260)

        # 学习方法选择
        learning_method = tk.StringVar()
        methods = ('Dydamic Programming', 'QLearning', 'SARSA', 'MonteCarlo', '3STEP-TD', 'Sarsa(λ)') # 下拉列表的值
        method_choosen = ttk.Combobox(window, width=12, textvariable=learning_method, values=methods, state='readonly')
        method_choosen.current(0)
        method_choosen.place(x=10, y=300)
        method_choosen.bind('<<ComboboxSelected>>', self.mothod_selected)

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
        self.method_choosen = method_choosen
        self.qtext = qtext
        self.qtext_showing = None
        self.path_lines = []
        self.path_state = None
        self.rl = None

    def mothod_selected(self, event):
        self.method_choosen.selection_clear()
        method = self.learning_method.get()
        if method == 'Dydamic Programming' and not self.started:
            self.canvas.itemconfig(self.rec, state=tk.HIDDEN)
        else:
            self.canvas.itemconfig(self.rec, state=tk.NORMAL)

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
        print('period:', self.period())

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
            self.canvas.itemconfig(self.values[i][j], text=str(round(v,2)).replace('0.','.') if 0 < np.abs(v) < 1 else str(int(v)))
            self.canvas.itemconfig(self.grid[i][j], fill=self.color(v))

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
            if method == 'Dydamic Programming':
                self.canvas.itemconfig(self.rec, state=tk.HIDDEN)
                self.rl = DP(self)
            elif method == 'MonteCarlo':
                self.rl = MonteCarlo(self)
            elif method == 'QLearning':
                self.rl = QLearning(self)
            elif method == 'TDLearning':
                self.rl = TDLearning(self)
            elif method == '3STEP-TD':
                self.rl = TDLearning(self, nstep=3)
            elif method == 'SARSA':
                self.rl = SARSA(self)
            elif method == 'Sarsa(λ)':
                self.rl = SarsaLambda(self)
            self.rl.start()
            self.start_text.set('stop')
        elif self.start_text.get() == 'stop':
            self.stop()

    def stop(self):
        self.started(False)
        self.event.set()
        self.start_text.set('start')

    def reset(self):
        for i in range(5):
            for j in range(5):
                self.canvas.itemconfig(self.values[i][j], text='0' if (i,j) not in self.terminals else '')
                self.canvas.itemconfig(self.grid[i][j], fill='#EEEEEE')
                for qtext in self.qtext[i][j].values():
                    self.canvas.itemconfig(qtext, text='', fill='#A020F0', state=tk.HIDDEN)
        self.move_to((0,0))
        self.pause_text.set('pause')
        self.print_step(0)
        self.print_episode(0)
        self.delete_path()


    def print_episode(self, episode):
        self.episode_text.set('episode: ' + str(episode))

    def print_step(self, step):
        self.step_text.set('step: ' + str(step))

    def update_qtext(self, updated, text_state=tk.HIDDEN):
        """
        更新状态的Q值
        """
        for (i,j), qtable in updated:
            qtext = self.qtext[i][j]
            maxq = np.max(qtable)
            for a,q in qtable.items():
                text = qtext[a]
                q_str = str(round(qtable[a],2)).replace('0.', '.') if 0 < np.abs(q) < 1 else str(int(q))
                color = '#FF00FF' if q == maxq and q != 0 else ('#CD3278' if q < 0 else '#A020F0')
                self.canvas.itemconfig(text, text=q_str, fill=color, state=text_state)

    def show_q(self, state):
        """
        显示状态的Q值
        """
        # print('show_q:', state)
        if state == self.qtext_showing or self.rl is None or isinstance(self.rl, DP):
            return
        self.hide_q()
        qtable = self.rl.state_qtable(state)
        self.update_qtext([(state, qtable)], text_state=tk.NORMAL)
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
            self.canvas.itemconfig(text, state=text_state)

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
        """
        绘制或清除路径
        """
        if self.rl is None:
            return

        if state == self.path_state:
            self.delete_path()
            return

        self.delete_path()
        path = self.rl.best_path(state)
        # print(path)

        for begin, end in zip(path[:-1], path[1:]):
            self.path_lines.append(self.canvas.create_line(*self.state_position(begin), *self.state_position(end), fill='#FFD39B', width=2))

        self.path_state = state

    def delete_path(self):
        for line in self.path_lines:
            self.canvas.delete(line)
        self.path_state = None

    def color(self, value):
        if value > 1:
            value = 1 - value / 9
        if value >= 0:
            c = int(255 * (1 - np.abs(value)) * self.color_scale)
            if c > 255:
                c = 255
            c = '%02x' % c
            # 正值为绿色
            rgb = '#' + c + 'ff' + c
        else:
            # 负值为红色
            c = int(255 * (1 + value) / self.color_scale)
            if c > 255:
                c = 255
            c = '%02x' % c
            rgb = '#ff' + c + c
        # print(value, rgb)
        return rgb


class RL:
    def __init__(self, maze, gamma=0.9, epsilon=0.6, alpha=0.5):
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
        self.updated = set()
        self.min_q_update_print = 1e-4    # 显示更新的最小增量(因显示更新比较耗时)

    def _state_q_init(self, state):
        actions = self.maze.state_actions(state)
        return pd.Series(data=np.zeros(len(actions)), index=actions)

    def move(self):
        action = self.epsilon_greedy()
        reward, next_state = self.take_action(action)
        return action, reward, next_state

    def take_action(self, action):
        next_state = self.next_state(action)
        reward = self.reward(action)
        self.move_to(next_state)
        return reward, next_state

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

    def update_q(self, state, action, q_target, step='avg'):
        """
        用现实的q来更新qtable中估计的Q值
        :param q_target: q现实
        :param step: alpha = 1/N if avg or self.alpha if fixed
        """
        if step == 'fixed':
            alpha = self.alpha
        elif step == 'avg':
            # 取平均值
            ntable = self.state_ntable(state)
            ntable[action] += 1
            alpha = 1 / ntable[action]
        q_eval = self.q(state, action)
        q_eval_new = q_eval + alpha * (q_target - q_eval)
        self._update_q(state, action, q_eval_new)

        # 如果更新量太小且已经更新过，不需要显示更新
        if np.abs(q_eval_new - q_eval) > self.min_q_update_print or state not in self.updated:
            self.print_updates([state])

        self.updated.add(state)

    def _update_q(self, state, action, q):
        self.state_qtable(state)[action] = q

    def epsilon_greedy(self):
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

    e_greedy = epsilon_greedy

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
            ending = False
            while self.started() and not ending:
                ending = self._learning()
        finally:
            self.move_to((0, 0))
            self.maze.stop()
            if self.closed():
                self.maze.quit()
        print('... Learning Ended ...')

    def _learning(self):
        raise NotImplementedError

    def best_path(self, state, maxlen=12):
        """
        获取某个状态的最优路径，路径最长maxlen个
        """
        path = [state]
        while len(path) < maxlen and path[-1] not in self.terminals:
            next_state = self.next_state(self.pi_star(state), state)
            path.append(next_state)
            state = next_state
        return path

    def print_updates(self, updates):
        update_values = [(state, self.maxq(state)) for state in updates]
        self.maze.print_value(update_values)

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

    def wait(self):
        self.maze.event.wait()


class DP(RL):
    def __init__(self, maze):
        RL.__init__(self, maze)
        # 每个状态保存到目标的最短距离和相应路径
        path = [[(999, '') for j in range(5)] for i in range(5)]
        i, j = self.goal[0]
        path[i][j] = (0, 'END') # 目标状态的距离和路径
        self.path = path
        maze.color_scale = 1.5

    def _state_q_init(self, state):
        """
        把q值设为''以免显示0
        """
        actions = self.maze.state_actions(state)
        return pd.Series(data=[''] * len(actions), index=actions)

    def distance(self, state1, state2):
        """
        两个相邻state之间的距离
        如果其中有一个是陷井，距离设为一个很大的值，其它相邻状态间的距离均为1
        """
        assert np.abs(np.subtract(state1,state2).sum()) == 1, 'states are not neighbors'

        if state1 in self.traps or state2 in self.traps:
            return 99999
        else:
            return 1

    def neighbors(self, state):
        """
        取一个状态的邻近状态
        """
        return [self.next_state(a, state) for a in self.maze.state_actions(state)]

    @staticmethod
    def neighbor_action(state1, state2):
        """
        从state1到state2需要采取什么动作
        """
        disi, disj = np.subtract(state1, state2)
        if disi == -1:
            return 'DOWN'
        elif disi == 1:
            return 'UP'
        elif disj == 1:
            return 'LEFT'
        elif disj == -1:
            return 'RIGHT'
        else:
            raise Exception('state1 state2 are not neighbors!')

    def state_path(self, state):
        i,j = state
        return self.path[i][j][1]

    def state_dist(self, state):
        i, j = state
        return self.path[i][j][0]

    def update_path(self, state, dist, path):
        i,j = state
        self.path[i][j] = (dist, path)

    def print_updates(self, updates):
        update_values = [(state, self.state_dist(state)) for state in updates]
        self.maze.print_value(update_values)

    def best_path(self, state, maxlen=10):
        """
        获取某个状态的最优路径，路径最长maxlen个
        """
        path = [state]
        while self.state_path(state) != '' and state not in self.terminals:
            state = self.next_state(self.state_path(state), state)
            path.append(state)
        return path

    def _learning(self):
        to_eval = self.goal.copy()  # 待评估状态
        while len(to_eval) > 0:
            self.wait()
            state = to_eval.pop(0)
            dist0 = self.state_dist(state)
            neighbors = self.neighbors(state)
            # print('eval:', state, neighbors)
            for neighbor in neighbors:
                self.wait()
                dist = self.distance(neighbor, state) + dist0
                if dist < self.state_dist(neighbor):
                    self.update_path(neighbor, dist, self.neighbor_action(neighbor, state))
                    self.print_updates([neighbor])
                    to_eval.append(neighbor)
                    self.wait_period()
        # self.print_path()
        return True

    def print_path(self):
        for p in self.path:
            print(p)


class QLearning(RL):
    """
    倒序更新的QLearning
    """
    def __init__(self, maze):
        RL.__init__(self, maze)
        maze.color_scale = 1.5

    def _learning(self):
        traces = self.simulate()
        traces.reverse()
        updates = []
        for state, action, reward, next_state in traces:
            q = reward + self.gamma * self.maxq(next_state)
            self.update_q(state, action, q, step='fixed')
            updates.append(state)
            # print(state, action, q)
        # self.print_updates(updates)
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


class MonteCarlo(RL):
    """
    使用Monte-Carlo方法进行学习
    """
    def __init__(self, maze):
        RL.__init__(self, maze)
        maze.color_scale = 1.15

    def _learning(self):
        traces = self.simulate()
        traces.reverse()
        q = 0
        for state, action, reward, next_state in traces:
            q = reward + self.gamma * q
            self.update_q(state, action, q)
            # print(state, action, q)
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
        self.steps = self.steps[1:]

    def learning_to_terminal(self):
        """
        走到终止态时使用存储的步骤进行学习
        """
        while len(self.steps) > 0:
            self.learning_nsteps()

    def _learning(self):
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


class SarsaLambda(RL):
    def __init__(self, maze, lambda_=0.9):
        RL.__init__(self, maze)
        self.lambda_ = lambda_
        self.eligibility_trace = [[self._state_q_init((i,j)) for j in range(5)] for i in range(5)]

    def state_trace(self, state):
        i,j = state
        return self.eligibility_trace[i][j]

    def trace_iter(self):
        return (((i,j),self.eligibility_trace[i][j]) for i in range(5) for j in range(5))

    def increase_eligibility_trace(self, state, action):
        self.state_trace(state)[action] = 1

    def discount_eligibility_trace(self):
        """
        将所有eligibility_trace的值用gamma * lambda_进行折算
        """
        for s,e in self.trace_iter():
            e *= self.gamma * self.lambda_

    def e(self, state, action):
        return self.state_trace(state)[action]

    def update_q(self, state, action, delta):
        # 取平均值
        ntable = self.state_ntable(state)
        ntable[action] += 1
        alpha = 1 / ntable[action]
        q = self.q(state, action)
        q = q + alpha * delta
        self._update_q(state, action, q)
        # 如果更新量太小且已经更新过，不需要显示更新
        if np.abs(alpha * delta) > self.min_q_update_print or state not in self.updated:
            start = time.time()
            self.print_updates([state])
            print_use_time(start, 'print_updates')
        self.updated.add(state)

    def update_qtable(self, delta):
        for state,trace in self.trace_iter():
            for action, e in trace.items():
                if e > 0:
                    self.update_q(state, action, delta * e)

    def _learning(self):
        action = self.epsilon_greedy()
        while  self.started() and self.state not in self.terminals:
            self.maze.event.wait()
            self.wait_period()
            state = self.state
            reward, next_state = self.take_action(action)
            self.increase_eligibility_trace(state, action)

            next_action = self.epsilon_greedy()
            q_target = reward + self.gamma * self.q(next_state, next_action)

            q_eval = self.q(state, action)
            delta = q_target - q_eval

            start = time.time()
            self.update_qtable(delta)
            print_use_time(start, 'update_qtable', min_time=10)

            self.discount_eligibility_trace()
            self.update_step()
            action = next_action
        self.update_episode()
        self.reset_eligibility_trace()
        self.wait_period(1.5)
        self.random_init_state()

    def reset_eligibility_trace(self):
        for s,e in self.trace_iter():
            e[:] = 0


class SARSA(RL):
    """
    等同于TD(0)或1 step TD
    """
    def __init__(self, maze):
        RL.__init__(self, maze, alpha=0.1)

    def _learning(self):
        action = self.e_greedy()
        while  self.started() and self.state not in self.terminals:
            self.maze.event.wait()
            state = self.state
            reward, next_state = self.take_action(action)
            next_action = self.e_greedy()
            q = reward + self.gamma * self.q(next_state, next_action)
            self.update_q(state, action, q)
            # self.maze.print_message(', '.join(map(str, (traces[-1]))))
            self.update_step()
            action = next_action
            self.wait_period()
        self.update_episode()
        self.wait_period(0.5)
        self.random_init_state()
        self.wait_period()


class DQN:
    pass

def print_use_time(start, info, min_time=1):
    use_time = int((time.time() - start) * 1000)
    if use_time > min_time:
        print(info, 'use:', use_time)

if __name__ == '__main__':
    mz = Maze(traps=[(2,1), (1,3)], goal=[(3,3)])
    # rl = QLearning(maze=mz)
    # rl = TDLearning(maze=mz)
    # rl.start()
    tk.mainloop()
    # DP(mz).learning()
    print('*********** ended ***********')
