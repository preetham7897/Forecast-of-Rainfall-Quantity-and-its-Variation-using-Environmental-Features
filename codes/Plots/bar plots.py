import matplotlib.pyplot as plt

ch_nm_mlr = [0.00441231, 0.066284987, 0.046238113, 0.030898223]
ch_pm_mlr = [0.741173057, 0.740627298]
ch_nm_svr = [0.004826504, 0.069365102, 0.052557173, 0.041867723]
ch_pm_svr = [0.734216348, 0.71612015]
ch_nm_dtr = [0.004675213, 0.068235318, 0.045567168, 0.029954508]
ch_pm_dtr = [0.725803896, 0.725115665]
ch_nm_pr = [0.004045785, 0.063463261, 0.043713317, 0.029171148]
ch_pm_pr = [0.762640385, 0.762135598]

r1_pm = [0, 1, 2, 3]
r2_pm = [0.2, 1.2, 2.2, 3.2]
r3_pm = [0.4, 1.4, 2.4, 3.4]
r4_pm = [0.6, 1.6, 2.6, 3.6]

plt.bar(r1_pm, ch_nm_mlr, color='red', width=0.2, edgecolor='white', label='MLR')
plt.bar(r2_pm, ch_nm_svr, color='blue', width=0.2, edgecolor='white', label='SVR')
plt.bar(r3_pm, ch_nm_dtr, color='green', width=0.2, edgecolor='white', label='DTR')
plt.bar(r4_pm, ch_nm_pr, color='cyan', width=0.2, edgecolor='white', label='PR')
        
plt.xlabel('Error Measures')
plt.ylabel('Error Values')
plt.xticks([0.3, 1.3, 2.3, 3.3], ['MSE', 'RMSE', 'MAE', 'MDAE']) 
plt.legend(loc='upper right')
plt.show()