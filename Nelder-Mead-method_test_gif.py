import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy import sin
import os


n = int(input('最佳化參數個數 n = '))

if n == 1:
    
    def objective_function(x):
        return sin(x) + sin((10.0 / 3.0) * x)    # 1D函數，可自行更換
    
elif n == 2:
    
    def objective_function(params):
        x, y = params
        return (x**2 + y - 11)**2 + (x + y**2 -7)**2    # 2D函數，可自行更換
    


def simplex_method_calculate_parameters():
    np.set_printoptions(precision=4, suppress=True)
    
    PP0 = np.zeros((n + 1, n))    # 儲存排序前
    PP = np.zeros((n + 1, n))     # 儲存排序後
    F = np.zeros(n + 1)

    # 新增用來保存每次迭代結果的列表
    all_PP = []
    all_F = []
    
    if n == 1:      
        for i in range(1, n + 2):
            for j in range(1, n + 1):
                PP0[i - 1, j - 1] = float(input(f'第 {i} 組參數向量的第 {j} 個元素 P({i}, {j}) = '))
            F[i - 1] = float(objective_function(PP0[i - 1, j - 1]))    
            
    elif n == 2: 
        for i in range(1, n + 2):
            params = []
            for j in range(1, n + 1):
                val = float(input(f'第 {i} 組參數向量的第 {j} 個元素 P({i}, {j}) = '))        
                PP0[i - 1, j - 1] = val         
                params.append(PP0[i - 1, j - 1])                  
            F[i - 1] = float(objective_function(params))
            
    else:   
        return print('圖片範例僅1D或2D函數')
        

        
    # 設定參數
    MaxIter = 100    # 最大迭代上限
    Tol_dJ = 1e-4
    Tol_norm_dP = 1e-4
    k = 1
    dJ = np.inf
    norm_dP = np.inf


    while k <= MaxIter and dJ > Tol_dJ and norm_dP > Tol_norm_dP:
        # 排序
        F_index = np.argsort(F)
        F = np.sort(F)
        PP = PP0[F_index]

        # 保存每次迭代結果
        all_PP.append(PP.copy())
        all_F.append(F.copy())

        J0 = F[0]    # min
        Jn = F[n]    # max

        dJ = abs(Jn - J0)
        norm_dP = np.linalg.norm(PP[n] - PP[0])

        P_ = np.mean(PP[:n], axis=0)        # 計算中軸點
        Pn = PP[n]
        Found_Pnew = 0
        
        Pr = P_ + P_ - Pn                   # 反射動作
        Jr = objective_function(Pr)

        if Jr < J0:                         # 擴展動作
            Pe = P_ + 2 * (P_ - Pn)
            Je = objective_function(Pe)

            if Je < Jr:
                Pnew = Pe
                Jnew = Je
                Found_Pnew = 1
            else:
                Pnew = Pr
                Jnew = Jr
                Found_Pnew = 1

        elif Jr < Jn:                       # 外部收縮動作
            Pc = P_ + 0.5 * (P_ - Pn)
            Jc = objective_function(Pc)

            if Jc <= Jr:
                Pnew = Pc
                Jnew = Jc
                Found_Pnew = 1
            else:
                Pnew = Pr
                Jnew = Jr
                Found_Pnew = 1

        else:                               # 內部收縮動作
            Pcc = P_ - 0.5 * (P_ - Pn)
            Jcc = objective_function(Pcc)

            if Jcc < Jn:
                Pnew = Pcc
                Jnew = Jcc
                Found_Pnew = 1

        if Found_Pnew == 0:                 # 多維收縮動作
            for i in range(1, n + 1):
                PP[i] = 0.5 * (PP[0] + PP[i])

                P_shrink = PP[i]
                F[i] = objective_function(P_shrink)

        else:
            PP[n] = Pnew
            F[n] = Jnew

        PP0 = PP
        
        P = PP[0]
        J = F[0]
        
        if n == 1:
            
            # 繪製目標1D函數圖形
            x_values = np.linspace(-3, 10, 1000)    # 可調節目標函數範圍
            
            y_values = objective_function(x_values)
            plt.plot(x_values, y_values, color='blue')
            
            # 繪製每次迭代的結果
            for i in range(len(all_PP)):
                plt.scatter(all_PP[i], all_F[i], color='limegreen', marker='o')
                plt.plot(all_PP[i], all_F[i], color='lightcoral', linestyle='-', linewidth=2)
            plt.savefig(f'1D_iteration_{k}.png')
            
        elif n == 2:
          
            # 繪製目標2D函數圖形
            x_range = np.linspace(-5, 5, 100)
            y_range = np.linspace(-5, 5, 100)    # 可調節目標函數範圍
            
            X, Y = np.meshgrid(x_range, y_range)
            Z = objective_function([X, Y])
            plt.contour(X, Y, Z, levels=30, cmap='viridis')
            
            # 繪製每次迭代的結果
            for i in range(len(all_PP)):
                x_values, y_values = zip(*all_PP[i])
                plt.scatter(x_values, y_values, color='limegreen', marker='o')
                plt.plot(x_values, y_values, color='lightcoral', linestyle='-', linewidth=2)
            plt.savefig(f'2D_iteration_{k}.png')
            
            
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Objective Function Contour Plot')
        plt.close()
        
        
        k += 1
        
        
    print(f'最佳參數向量: {P}')
    print(f'最佳目標函數結果: {J}')
    print(f'總迭代次數: {k}')
    print(f'dJ: {dJ}')
    print(f'Norm_dP: {norm_dP}')
        


    # 創建 GIF
    if n == 1:
        images = [imageio.imread(f'1D_iteration_{i}.png') for i in range(1, k)]
        imageio.mimsave('1D_iteration_result.gif', images, fps=1.5)
        for i in range(1, k):
            os.remove(f'1D_iteration_{i}.png')
    elif n == 2:
        images = [imageio.imread(f'2D_iteration_{i}.png') for i in range(1, k)]
        imageio.mimsave('2D_iteration_result.gif', images, fps=1.5)
        for i in range(1, k):
            os.remove(f'2D_iteration_{i}.png')


simplex_method_calculate_parameters()