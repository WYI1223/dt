#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import time
import random
import pandas as pd
from base.SelfImproving.training import training_phase
from base.SelfImproving.Operation_simplifed import operation_phase
from base.SelfImproving.PointDistribution2 import generate_clustered_points_superfast
from base.SelfImproving.Delaunay_face_routing_simplified import compute_delaunay
from base.DelunayClassic import compute_delaunay as compute_delaunay_classic
from base.DCEL.vertex import Vertex as Point

def run_benchmarks(output_file='benchmark_results-1.csv'):
    # 参数空间
    ns      = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    ks      = [3, 4, 5]
    layouts = ["x_split", "grid"]

    # 写入表头
    cols = [
        'n', 'm', 'k', 'layout',
        't_train', 't_op', 't_gen', 't_input', 't_classic'
    ]
    pd.DataFrame(columns=cols).to_csv(output_file, index=False)

    for n in ns:
        ms = [n // 100, n // 10, n // 5]
        if  n >= 1000:
            ms = [n//1000,n//500,n//100,n//50]
            if n>= 4000:
                ms = [n//1000, n//500]
        ms = [m for m in ms if m > 0]

        for m, k, layout in itertools.product(ms, ks, layouts):
            random.seed(42)

            # 生成训练样本
            train_samples = [
                generate_clustered_points_superfast(
                    n, k,
                    x_range=(5.01, 15),
                    y_range=(0.01, 7.5),
                    std_dev=0.4,
                    layout=layout
                )
                for _ in range(m)
            ]

            # 训练阶段
            t0 = time.perf_counter()
            state = training_phase(train_samples, n, k)
            t_train = time.perf_counter() - t0

            # 生成测试实例
            I_new = generate_clustered_points_superfast(
                n, k,
                x_range=(5.01, 15),
                y_range=(0.01, 7.5),
                std_dev=0.4,
                layout=layout
            )

            # operation_phase
            t0 = time.perf_counter()
            results = operation_phase(I_new, state)
            t_op = time.perf_counter() - t0

            # Delaunay generator
            t0 = time.perf_counter()
            _ = compute_delaunay(results)
            t_gen = time.perf_counter() - t0

            # Delaunay on 原始点
            I_vertices = [Point(x, y) for x, y in I_new]
            t0 = time.perf_counter()
            _ = compute_delaunay(I_vertices)
            t_input = time.perf_counter() - t0

            # # 经典算法：仅当 n <= 1000 时才测时
            # if n <= 1000:
            #     t0 = time.perf_counter()
            #     _ = compute_delaunay_classic(I_vertices)
            #     t_classic = time.perf_counter() - t0
            # else:
            #     t_classic = None  # 跳过
            t_classic = None
            # 写入结果（追加模式）
            row = {
                'n': n, 'm': m, 'k': k, 'layout': layout,
                't_train': t_train,
                't_op':    t_op,
                't_gen':   t_gen,
                't_input': t_input,
                't_classic': t_classic
            }
            pd.DataFrame([row]).to_csv(
                output_file,
                mode='a',
                header=False,
                index=False
            )

            # 打印到控制台
            classic_msg = f"{t_classic:.3f}s" if t_classic is not None else "skipped"
            print(f"Done: n={n}, m={m}, k={k}, layout={layout}   "
                  f"t_train={t_train:.3f}s, t_op={t_op:.3f}s, "
                  f"t_gen={t_gen:.3f}s, t_input={t_input:.3f}s, "
                  f"t_classic={classic_msg}")

if __name__ == "__main__":
    run_benchmarks()
