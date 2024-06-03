import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot(member_file, non_member_file):
    with open(member_file, 'rb') as f:
        member_loss = pickle.load(f)
        
    with open(non_member_file, 'rb') as f:
        non_member_loss = pickle.load(f)
        
    print("member loss length: ", len(member_loss))
    print("non_member_loss: ", len(non_member_loss))
    
    # 计算统计量
    mean_non_member = np.mean(non_member_loss)
    var_non_member = np.var(non_member_loss)
    
    mean_member = np.mean(member_loss)
    var_member = np.var(member_loss)

    # 绘制直方图
    bins = np.linspace(0, 15, 100)
    plt.hist(non_member_loss, bins, alpha=0.5, label='non-member', color='blue', density=True)
    plt.hist(member_loss, bins, alpha=0.5, label='member', color='green', density=True)

    # 添加均值的虚线
    plt.axvline(mean_non_member, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(mean_member, color='green', linestyle='dashed', linewidth=1)

    # 添加统计信息
    plt.text(12, 0.25, f'$\\mathbb{{E}}[\\ell]_{{\\mathrm{{mem}}}}={mean_member:.2f}$\n$\\mathrm{{Var}}[\\ell]_{{\\mathrm{{mem}}}}={var_member:.2f}$', fontsize=8, color=(0.3, 0.3, 0.3, 1.0))
    plt.text(12, 0.2, f'$\\mathbb{{E}}[\\ell]_{{\\mathrm{{non}}}}={mean_non_member:.2f}$\n$\\mathrm{{Var}}[\\ell]_{{\\mathrm{{non}}}}={var_non_member:.2f}$', fontsize=8, color=(0.3, 0.3, 0.3, 1.0))

    # 添加标签、图例和网格
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.title('Loss Distribution for Members and Non-members')
    plt.grid(False)

    plt.savefig('./results/loss_smoothl1_tmp.png', dpi=1600)
    
if __name__ == '__main__':
    member_file  = 'membersmtmp_sumloss.pkl'
    non_member_file = 'non_membersmtmp_sumloss.pkl'
    plot(member_file, non_member_file)