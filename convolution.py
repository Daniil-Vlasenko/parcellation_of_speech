import audio_processing_Daniil as apD
import numpy as np
import glob


H_path = ("../data/H/S.npy", "../data/H/T.npy")


# Среднее значение частотных каналов вызванное началам слогов
def R_calculation(I_INP, Y):
    Y_for_R = np.zeros(Y.shape[0], dtype=int)
    for count, value in enumerate(Y):
        if value == 1:
            for i in range(min(count, 61)):
                Y_for_R[count - i] = 1

    n = len(I_INP)

    results = []
    for u in range(6):
        result = []
        # sum_Y_for_R = sum(Y_for_R)
        for i in range(I_INP.shape[1]):
            result.append(np.dot(I_INP[:n - u, i], Y_for_R[u:]) / len(Y_for_R[u:]))
        results.append(result)
    R = np.array(results)
    print(f'{R.shape=}')
    return R


# Корреляция между каналами w и xi, с задержками u и v
def M_calculation(I_INP):
    count = 0
    result = np.zeros((32, 32, 6, 6))
    for w in range(32):
        for xi in range(32):
            for u in range(6):
                for v in range(6):
                    dif = u - v
                    if dif > 0:
                        result[w, xi, u, v] = np.cov(I_INP[:-abs(dif), w], I_INP[abs(dif):, xi])[0, 1]
                    elif dif < 0:
                        result[w, xi, u, v] = np.cov(I_INP[abs(dif):, w], I_INP[:-abs(dif), xi])[0, 1]
                    else:
                        result[w, xi, u, v] = np.cov(I_INP[:, w], I_INP[:, xi])[0, 1]

                    print(f"M:{count}")
                    count += 1

    return result


# Условие остановки обучения свертки (не понял условие из статьи), остановка происходит, когда евклидово расстояние
# между S и S_last и между T и T_last будет меньше threshold
def is_stop(S, T, S_last, T_last, threshold=10e-4):
    print(f"is_stop: {np.linalg.norm(S - S_last)},\t {np.linalg.norm(T - T_last)}")
    return np.linalg.norm(S - S_last) < threshold and np.linalg.norm(T - T_last) < threshold


# Обучение ядра свертки, сохранение компонент ядра в отдельных файлах
def H_calculation(I_INP_path, Y_path, H_path):
    I_INP = np.load(I_INP_path)
    Y = np.load(Y_path)
    R = R_calculation(I_INP, Y)
    M = M_calculation(I_INP)
    S = np.ones(32)
    T = np.ones(6)
    S_last = np.ones(32)
    T_last = np.ones(6)

    count = 0
    while True:
        S_numerator = T.T @ R
        S_denominator = 0
        for u in range(6):
            for v in range(6):
                S_denominator += T[u] * T[v] * M[:, :, u, v]
        S_last = S.copy()
        S = (S_numerator @ np.linalg.inv(S_denominator)).T

        T_numerator = R @ S
        T_denominator = 0
        for w in range(32):
            for xi in range(32):
                T_denominator += S[w] * S[xi] * M[w, xi, :, :]
        T_last = T.copy()
        T = np.linalg.inv(T_denominator) @ T_numerator

        print(f"convolution_learning: {count}")
        count += 1
        if is_stop(S, T, S_last, T_last):
            break

    np.save(H_path[0], S)
    np.save(H_path[1], T)


# Свертка одного предложения
def convolution(I_INP_path, H_path):
    I_INP = np.load(I_INP_path)
    S = np.load(H_path[0])
    T = np.load(H_path[1])

    def H(w, u):
        return S[w] * T[u]

    result = np.zeros(I_INP.shape[0])
    for t in range(5, I_INP.shape[0]):
        for w in range(32):
            for u in range(6):
                result[t] = H(w, u) * I_INP[t - u, w]

    return result




