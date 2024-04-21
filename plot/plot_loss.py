import matplotlib.pyplot as plt
import pickle

def plot(member_file, non_member_file):
    with open(member_file, 'rb') as f:
        member_loss = pickle.load(f)
        
    with open(non_member_file, 'rb') as f:
        membernon_member_loss = pickle.load(f)
        
    print("member loss length: ", len(member_loss))
    print("membernon_member_loss: ", len(membernon_member_loss))
    
    
    plt.hist(member_loss, bins=50, color='blue', alpha=0.7, label='member loss')
    plt.hist(membernon_member_loss, bins=50, color='green', alpha=0.7, label='non_member loss')
    plt.title('loss Distribution')
    plt.xlabel('loss')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('./results/loss.png', dpi=1600)
    
if __name__ == '__main__':
    member_file  = 'member_loss.pkl'
    non_member_file = 'non_member_loss.pkl'
    plot(member_file, non_member_file)