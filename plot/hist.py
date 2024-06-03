import matplotlib.pyplot as plt
import numpy as np

# 设置样式
# plt.style.use('seaborn-darkgrid')

def plot():
    # 模拟数据
    np.random.seed(0)
    non_member_losses = np.random.gamma(shape=2.0, scale=1.0, size=1000)
    member_losses = np.random.gamma(shape=2.0, scale=0.5, size=1000)

    # 计算统计量
    mean_non_member = np.mean(non_member_losses)
    var_non_member = np.var(non_member_losses)
    mean_member = np.mean(member_losses)
    var_member = np.var(member_losses)

    # 绘制直方图
    bins = np.linspace(0, 25, 50)
    plt.hist(non_member_losses, bins, alpha=0.5, label='non-member', color='blue', density=True)
    plt.hist(member_losses, bins, alpha=0.5, label='member', color='green', density=True)

    # 添加均值的虚线
    plt.axvline(mean_non_member, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(mean_member, color='green', linestyle='dashed', linewidth=1)

    # 添加统计信息
    plt.text(15, 0.3, f'$\\mathbb{{E}}[\\ell]_{{\\mathrm{{mem}}}}={mean_member:.2f}$\n$\\mathrm{{Var}}[\\ell]_{{\\mathrm{{mem}}}}={var_member:.2e}$', fontsize=8, color=(0.3, 0.3, 0.3, 1.0))
    plt.text(15, 0.2, f'$\\mathbb{{E}}[\\ell]_{{\\mathrm{{non}}}}={mean_non_member:.2f}$\n$\\mathrm{{Var}}[\\ell]_{{\\mathrm{{non}}}}={var_non_member:.2f}$', fontsize=8, color=(0.3, 0.3, 0.3, 1.0))

    # 添加标签、图例和网格
    plt.xlabel('Loss')
    plt.ylabel('Normalized Frequency')
    plt.legend(loc='best')
    plt.title('Loss Distribution for Members and Non-members')
    plt.grid(False)



# 保存图形
plt.savefig('loss_distribution.png', dpi=300)
 
if __name__ == '__main__':
    member_file  = 'member_loss.pkl'
    non_member_file = 'non_member_loss.pkl'
    plot(member_file, non_member_file)