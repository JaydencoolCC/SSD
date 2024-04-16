import pickle
import matplotlib.pyplot as plt

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_data(data):
    results = {} # key: image_id, value: [[score, IOU], [score, IOU], ...]
    for d in data:
        image_id = d['image_id']
        if image_id in results:
            value = [d['score'], d['IOU']]
            results[image_id].append(value)
        else:
            results[image_id] = [[d['score'], d['IOU']]] #score, IOU
    
    results = dict(sorted(results.items()))
    scores, IOU, FP, FN = [], [], [], []    
    for key, data in results.items():
        if(len(data)>20):
            print("data:`", len(data))
            print("key:", key)
            print(data)
        fp = 0
        fn = 0
        for val in data:

            iou = val[1]
            if iou == 0:
                fp += 1
                
            elif iou == -1:  # Fixed the syntax error here
                fn += 1
            else:
                scores.append(val[0])
                IOU.append(val[1])
        FP.append(fp)
        FN.append(fn)
        #print(fn)
    return [scores, IOU, FP, FN]        
                
def plot(train_result, test_result):
    trian_score, train_IOU, train_FP, train_FN = train_result
    test_score, test_IOU, test_FP, test_FN = test_result
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.hist(trian_score, bins=20, color='blue', alpha=0.7, label='Train Score')
    plt.hist(test_score, bins=20, color='red', alpha=0.7, label='Test Score')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 2)
    plt.hist(train_IOU, bins=20, color='blue', alpha=0.7, label='Train IOU')
    plt.hist(test_IOU, bins=20, color='red', alpha=0.7, label='Test IOU')
    plt.title('IOU Distribution')
    plt.xlabel('IOU')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 3)
    plt.hist(train_FP, bins=20, color='blue', alpha=0.7, label='Train FP')
    plt.hist(test_FP, bins=20, color='red', alpha=0.7, label='Test FP')
    plt.title('FP Distribution')
    plt.xlabel('FP')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 4)
    plt.hist(train_FN, bins=20, color='blue', alpha=0.7, label='Train FN')
    plt.hist(test_FN, bins=20, color='red', alpha=0.7, label='Test FN')
    plt.title('FN Distribution')
    plt.xlabel('FN')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('plot.png', dpi=1600)
    plt.show()
    
                
if __name__ == '__main__':
    train_file_path = r'D:\WorkSpace\GithubCode\MIA\SSD\train_dataset.pkl'
    train_data = load_pkl_file(train_file_path)
    train_result = compute_data(train_data)
    
    test_file_path = r'D:\WorkSpace\GithubCode\MIA\SSD\test_dataset.pkl'
    test_data = load_pkl_file(test_file_path)
    test_result = compute_data(test_data)
    
    plot(train_result, test_result)
    
    
        