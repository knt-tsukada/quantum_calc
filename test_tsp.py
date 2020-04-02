# cd /mnt/c/workspace/quantum_calc
# python3 test_tsp.py

from dimod import *
import numpy as np
import matplotlib.pyplot as plt
import random
import math 

from dwave_qbsolv import QBSolv

from dwave.system.samplers import *
from dwave.system.composites import EmbeddingComposite

### プロキシを通す ###
# https://qiita.com/shnchr/items/bbecc7cf984bf80aee27
# import os
# os.environ["http_proxy"] = "XXXXXX"
# os.environ["https_proxy"] = "XXXXXX"
##################

def gen_problem(size=4, is_random=False):
	# 地点Xと地点Yの間の経路長（コスト）
	pos = []
	d = {}

	if is_random == False:
		if size == 4:
			d = {(0,1): 3,
				(0,2): 20,
				(0,3): 3,
				(1,2): 5,
				(1,3): 20,
				(2,3): 4
			}
			# A -> B -> C -> D -> A が正解
			# この時の全経路は15
		elif size == 5:
			d = {(0,1): 3,
				(0,2): 20,
				(0,3): 20,
				(0,4): 4,
				(1,2): 2,
				(1,3): 20,
				(1,4): 20,
				(2,3): 3,
				(2,4): 20,
				(3,4): 4
			}
			# A -> B -> C -> D -> E -> A が正解
			# この時の全経路は16
	else:
		for i in range(size):
			# x = random.randint(0, 1000) / 100	# 0.00 ~ 10.00
			# y = random.randint(0, 1000) / 100	# 0.00 ~ 10.00
			x = (math.sin(math.pi * 2 *  i / size) + 1) * 5
			y = (math.cos(math.pi * 2 *  i / size) + 1) * 5
			pos.append((x, y, chr(ord('A')+i) ))

		for i in range(size):
			for j in range(i+1, size):
				P = pos[i]
				Q = pos[j]
				d[(i, j)] = ( (P[0] - Q[0])**2 + (P[1] - Q[1])**2 )**0.5
	return (pos, d)

def show_plots(pos, d):
	[plt.plot(x, y, marker='o') for (x, y, name) in pos]
	plt.show()

	# WSLのUbuntu18.04でmatplotlibを使えるようにするまで
	# http://ai-gaminglife.hatenablog.com/entry/2019/04/29/204841
	# 
	# 以下を入れる
	# > sudo apt install python3-tk
	# > sudo apt install tk-dev
	#
	# Windows側で、sourceForge から VcXsrvをDLする。
	#
	# 実行環境の修正
	# > sudo vim ~/.bashrc
	# 最後の行に以下を追加して保存する。
	# export DISPLAY=:0.0

def constraint_matrix(size):
	# 以下のような行列を作る。	
	# size==3 のとき：
	# [[-1, 2, 2],
	# [ 0, -1, 2],
	# [ 0, 0, -1]]

	# size==5 のとき：
	# [[-1, 2, 2, 2, 2],
	# [ 0, -1, 2, 2, 2],
	# [ 0, 0, -1, 2, 2],
	# [ 0, 0, 0, -1, 2],
	# [ 0, 0, 0, 0, -1]]

	tmp = (2 * np.triu(np.ones(size), k=1) - np.identity(size))
	return tmp

def show_solve(pos, path):
	xs = [pos[i][0] for i in path[1]]
	ys = [pos[i][1] for i in path[1]]

	xs.append(xs[0])	# 最後の地点と最初の地点を結ぶため
	ys.append(ys[0])

	plt.plot(xs, ys, marker='o', linestyle='--')
	plt.plot(xs[0], ys[0], color='red', marker='o', linestyle='--')
	for x, y, name in pos:
		plt.text(x, y, name)
	plt.show()


if __name__ == "__main__":
	SIZE=4
	pos, d = gen_problem(size=SIZE, is_random=False)
	# pos, d = gen_problem(size=SIZE, is_random=True)

	print(pos)
	print(d)
	# グラフ表示するときはコメントアウト外す
	# show_plots(pos, d)

	# 行、列の意味：[q_A0, q_B0, ..., q_D0, qA1, ..., q_C3, qD_3]
	# q_Xt: ステップtのときに、地点Xにいるかどうかを表すバイナリ変数。
	H = np.zeros((SIZE**2, SIZE**2))
	teisuu = 0

	# 係数が 1 だとうまく答えがでない。H1に比べてペナルティが弱いため？？
	a1 = 10
	a2 = 10

	# 目的関数 H1：経路を最小化する
	H1 = np.zeros((SIZE**2, SIZE**2))
	for alpha in range(SIZE):
		for beta in range(SIZE):
			if (alpha == beta):
				continue
			# print(alpha, beta)
			for t in range(SIZE):
				dist = 0
				if (alpha < beta):
					dist = d[(alpha, beta)]
				else:
					dist = d[(beta, alpha)]

				row = alpha + SIZE*t
				col = beta  + (SIZE*(t+1)) % (SIZE**2)
				if (row < col):
					H1[row, col] += dist
				else:
					H1[col, row] += dist	# 最後のステップから、0番目のステップに戻る場合。
											# QUBOの行列の下三角をすべて0にするために必要。
	print("H1:\n", H1, "\n")
	H += H1

	tmp = constraint_matrix(SIZE)

	# 目的関数 H2：各時点で訪れる地点は一つだけ
	H2 = np.zeros((SIZE**2, SIZE**2))
	for m in range(SIZE):
		for i in range(SIZE):
			for j in range(SIZE):
				H2[i + SIZE*m, j + SIZE*m] += tmp[i, j] * a1
	print("H2:\n", H2, "\n")
	H += H2
	teisuu += a1

	# 目的関数 H3：各地点には一回しか訪れない
	H3 = np.zeros((SIZE**2, SIZE**2))
	for m in range(SIZE):
		for i in range(SIZE):
			for j in range(SIZE):
				H3[SIZE*i + m, SIZE*j + m] += tmp[i, j] * a2
	print("H3:\n", H3, "\n")
	H += H3
	teisuu += a2

	# （おまけ、目的関数 H4：地点Aが一番目に来るようにする）
	# 式：H4 = a3 * (q_A0 - 1)^2 = a3 * (-q_A0^2 + 1)
	H4 = np.zeros((SIZE**2, SIZE**2))
	a3 = 20

	H4[0, 0] = -a3
	H += H4
	teisuu += a3

	print("H:\n", H, "\n")

	# QUBOモデルにする
	Q = H.astype(np.float32)

	b_model = BinaryQuadraticModel.from_numpy_matrix(Q, offset = teisuu)
	### サンプラー ###
	# r = ExactSolver().sample(b_model)

	# r = SimulatedAnnealingSampler().sample(b_model)

	# r = QBSolv().sample(b_model)

	# https://qiita.com/YuichiroMinato/items/57cb8504ab61930eb479
	endpoint	='https://cloud.dwavesys.com/sapi/'
	token		='YOUR_TOKEN'
	solver		='DW_2000Q_5'
	r = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token, solver=solver)).sample(b_model, num_reads=10)
	
	# r = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token, solver=solver)).sample_qubo(Q, num_reads=1000)

	##############
	paths = []
	for s, e in r.data(['sample','energy']):
		# print(s,'\t E = ', e)
		tmp = np.zeros((SIZE, SIZE))
		for i in range(SIZE**2):
			tmp[i // SIZE, i % SIZE] = s[i]
		print("  A, B, C, D, E, ...")
		print(tmp,' Energy = ', e, "\n")

		cost = 0.0
		path = []
		for m in tmp:
			try:
				path.append(m.tolist().index(1))
			except ValueError:
				print("ValueError occurs.")
				continue

		# 総経路長の計算
		try:
			for i in range(SIZE):
				p1 = path[i]
				p2 = path[(i+1)%SIZE]
				if (p1 < p2):
					cost += d[(p1, p2)]
				else:
					cost += d[(p2, p1)]
				 
			paths.append((cost, path))
		except IndexError:
			print("IndexError occurs. The result cannot be solved.")
			continue

		except KeyError:
			print("KeyError occurs. The result cannot be solved.")
			continue

	[print(p) for p in paths]


	# グラフ表示するときはコメントアウト外す
	# show_solve(pos, paths[0])
