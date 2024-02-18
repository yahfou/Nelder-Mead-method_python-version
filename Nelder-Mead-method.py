import numpy as np

def simplex_method_calculate_parameters():
    np.set_printoptions(precision=4, suppress=True)

    # Step 1
    n = int(input('最佳化參數個數 n = '))
    PP0 = np.zeros((n + 1, n))  # 排序前
    PP = np.zeros((n + 1, n))   # 排序後
    F = np.zeros(n + 1)

    for i in range(1, n + 2):
        for j in range(1, n + 1):
            PP0[i - 1, j - 1] = float(input(f'第 {i} 組參數向量的第 {j} 個元素 P({i}, {j}) = '))

        F[i - 1] = float(input(f'此組參數的目標函數結果 J({i}) = '))

    # Parameters
    MaxIter = 100
    Tol_dJ = 1e-4
    Tol_norm_dP = 1e-4
    k = 1
    dJ = np.inf
    norm_dP = np.inf

    while k <= MaxIter and dJ > Tol_dJ and norm_dP > Tol_norm_dP:
        # Sorting
        F_index = np.argsort(F)
        F = np.sort(F)
        PP = PP0[F_index]

        J0 = F[0]
        Jn = F[n]

        dJ = abs(Jn - J0)
        norm_dP = np.linalg.norm(PP[n] - PP[0])

        P_ = np.mean(PP[:n], axis=0)
        Pn = PP[n]
        Found_Pnew = 0

        print('==============================================')
        print(f'迭代次數 k = {k} ')
        print('新參數如下，輸入至目標函數取得結果後輸入至下方性能指標')
        Pr = P_ + P_ - Pn
        print(f'Pr = {Pr}')
        Jr = float(input('性能指標 Jr = '))
        print('----------------------------------------------')

        if Jr < J0:
            print('新參數如下，輸入至目標函數取得結果後輸入至下方性能指標')
            Pe = P_ + 2 * (P_ - Pn)
            print(f'Pe = {Pe}')
            Je = float(input('性能指標 Je = '))
            print('----------------------------------------------')

            if Je < Jr:
                Pnew = Pe
                Jnew = Je
                Found_Pnew = 1
            else:
                Pnew = Pr
                Jnew = Jr
                Found_Pnew = 1

        elif Jr < Jn:
            print('新參數如下，輸入至目標函數取得結果後輸入至下方性能指標')
            Pc = P_ + 0.5 * (P_ - Pn)
            print(f'Pc = {Pc}')
            Jc = float(input('性能指標 Jc = '))
            print('----------------------------------------------')

            if Jc <= Jr:
                Pnew = Pc
                Jnew = Jc
                Found_Pnew = 1
            else:
                Pnew = Pr
                Jnew = Jr
                Found_Pnew = 1

        else:
            print('新參數如下，輸入至目標函數取得結果後輸入至下方性能指標')
            Pcc = P_ - 0.5 * (P_ - Pn)
            print(f'Pcc = {Pcc}')
            Jcc = float(input('性能指標 Jcc = '))
            print('----------------------------------------------')

            if Jcc < Jn:
                Pnew = Pcc
                Jnew = Jcc
                Found_Pnew = 1

        if Found_Pnew == 0:
            for i in range(1, n + 1):
                PP[i] = 0.5 * (PP[0] + PP[i])
                print('新參數如下，輸入至目標函數取得結果後輸入至下方性能指標')
                P_shrink = PP[i]
                print(f'P_shrink = {P_shrink}')
                F[i] = float(input('性能指標 F = '))
                print('----------------------------------------------')
        else:
            PP[n] = Pnew
            F[n] = Jnew

        PP0 = PP
        k += 1

    # Sorting again
    F_index = np.argsort(F)
    F = np.sort(F)
    PP = PP0[F_index]

    P = PP[0]
    J = F[0]

    print(f'最佳參數向量: {P}')
    print(f'最佳目標函數結果: {J}')
    print(f'總迭代次數: {k}')
    print(f'dJ: {dJ}')
    print(f'Norm_dP: {norm_dP}')


# Run the function
simplex_method_calculate_parameters()