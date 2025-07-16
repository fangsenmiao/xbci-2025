from psychopy import visual, core, sound, event, data, logging
import numpy as np
import serial
import random
import os

# 确保中文正常显示
import matplotlib

matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 实验基本设置
n_trials = 240  # 总试次数量
instructions = ['上', '下', '左', '右', '停']  # 5种指令
trials_per_instruction_in_block = 2  # 每10个试次中每种指令出现的次数
instructions_per_block = 10  # 每个试次块的大小
trials_per_block = 5  # 每个循环的试次数量
blocks_per_break = 12  # 每完成多少个循环休息一次
break_duration = 30  # 休息时间（秒）

# 创建窗口
win = visual.Window(
    size=(1024, 768), fullscr=True, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

# 创建刺激对象
fixation = visual.TextStim(win=win, text='+', height=0.1, color='white')
instruction_text = visual.TextStim(win=win, text='', height=0.2, color='white')
break_text = visual.TextStim(win=win, text='请闭眼休息，30秒后继续...', height=0.15, color='white')
escape_text = visual.TextStim(win=win, text='按ESC键退出实验', height=0.07, pos=(0, -0.9), color='gray')

# 创建提示音
beep = sound.Sound(value=440, secs=0.2, stereo=True)

# 准备串口通信
try:
    ser = serial.Serial('COM4', 115200, timeout=1)
    print("串口COM4连接成功")
except:
    print("无法连接到串口COM4，将继续实验但不进行打标")
    ser = None

# 创建实验数据结构
trial_list = []

# 计算需要多少个10试次块
num_blocks = n_trials // instructions_per_block

for block in range(num_blocks):
    # 为当前块创建基础试次列表 (每种指令2次)
    block_trials = []
    for instruction in instructions:
        block_trials.extend([{'instruction': instruction}] * trials_per_instruction_in_block)

    # 随机打乱当前块内的试次顺序
    random.shuffle(block_trials)

    # 将当前块的试次添加到总试次列表
    trial_list.extend(block_trials)

# 创建实验流程
trials = data.TrialHandler(trialList=trial_list, nReps=1, method='sequential')


# 检查退出函数
def check_escape():
    if event.getKeys(keyList=["escape"]):
        return True
    return False


# 实验开始
for trial_num, trial in enumerate(trials):
    # 每完成60个试次，让被试休息30秒
    if trial_num > 0 and trial_num % 60 == 0:
        break_text.draw()
        escape_text.draw()
        win.flip()
        start_time = core.getTime()
        while core.getTime() - start_time < break_duration:
            if check_escape():
                break
        if check_escape():
            break

    # 提示音
    beep.play()
    escape_text.draw()
    win.flip()
    core.wait(0.2)
    if check_escape():
        break

    # 注视点
    fixation.draw()
    escape_text.draw()
    win.flip()
    start_time = core.getTime()
    while core.getTime() - start_time < 2.0:
        if check_escape():
            break
    if check_escape():
        break

    # 显示指令符号
    instruction_text.setText(trial['instruction'])
    instruction_text.draw()
    escape_text.draw()
    win.flip()
    core.wait(1.25)  # 指令符号显示1.25秒
    if check_escape():
        break

    # 开始想象阶段（从第3秒到第7秒）
    if ser:
        try:
            # 发送打标信息到串口
            marker = {'上': b'1', '下': b'2', '左': b'3', '右': b'4', '停': b'5'}
            ser.write(marker[trial['instruction']])
        except:
            print("串口打标失败")

    # 清空屏幕，进入想象阶段
    escape_text.draw()
    win.flip()
    start_time = core.getTime()
    while core.getTime() - start_time < 4.0:
        if check_escape():
            break
    if check_escape():
        break

    # 休息阶段
    escape_text.draw()
    win.flip()
    start_time = core.getTime()
    while core.getTime() - start_time < 3.0:
        if check_escape():
            break
    if check_escape():
        break

# 关闭串口连接
if ser:
    ser.close()

# 关闭窗口
win.close()

# 退出实验
core.quit()