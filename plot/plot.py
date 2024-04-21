import pickle
import matplotlib.pyplot as plt
import numpy as np
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
    print("size: ", len(results))
    results = dict(sorted(results.items()))
    scores, IOU, FP, FN = [], [], [], []    
    for key, data in results.items():
        fp = 0
        fn = 0
        for val in data:

            iou = val[1]
            if iou == 0:
                fp += 1
                
            elif iou == -1:  #
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
    bin =  max(train_FP) if max(train_FP) > max(test_FP) else max(test_FP)
    plt.hist(train_FP, bins=bin, color='blue', alpha=0.7, label='Train FP')
    plt.hist(test_FP, bins=bin, color='red', alpha=0.7, label='Test FP')
    plt.title('FP Distribution')
    plt.xlabel('FP')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, max(max(train_FP), max(test_FP)) + 1)) 
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 4)
    bin =  max(train_FN) if max(train_FN) > max(test_FN) else max(test_FN)
    plt.hist(train_FN, bins=bin, color='blue', alpha=0.7, label='Train FN')
    plt.hist(test_FN, bins=bin, color='red', alpha=0.7, label='Test FN')
    plt.title('FN Distribution')
    plt.xlabel('FN')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, max(max(train_FN), max(test_FN)) + 1)) 
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('plot.png', dpi=1600)
    plt.show()
    
def convert_data(data):
    results = {} # key: image_id, value: [[score, IOU], [score, IOU], ...]
    for d in data:
        image_id = d['image_id']
        if image_id in results:
            value = [d['score'], d['IOU']]
            results[image_id].append(value)
        else:
            results[image_id] = [[d['score'], d['IOU']]] #score, IOU
    
    results = dict(sorted(results.items()))
    return results

def plot2(train_result, test_result):
    train_con = []
    test_con = []
    for key, data in train_result.items():
        score = 0
        for val in data:
            score = score + 0.5 * val[0] + 0.5 * max(val[1], 0)
        train_con.append(score/len(data))
        
    for key, data in test_result.items():
        score = 0
        for val in data:
            score = score + 0.5 * val[0] + 0.5 * max(val[1], 0)
        test_con.append(score/len(data))
                
    plt.hist(train_con, bins=50, color='blue', alpha=0.7, label='Train Score')
    plt.hist(test_con, bins=50, color='red', alpha=0.7, label='Test Score')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('score.png', dpi=1600)
     
if __name__ == '__main__':
    
    # train_file_path ='./member_dataset.pkl'
    # train_data = load_pkl_file(train_file_path)
    # train_result = compute_data(train_data)
    
    # test_file_path = './non-member_dataset.pkl'
    # test_data = load_pkl_file(test_file_path)
    # test_result = compute_data(test_data)
    
    # plot(train_result, test_result)
    
    train_file_path ='./member_dataset.pkl'
    train_data = load_pkl_file(train_file_path)
    train_result = convert_data(train_data)
    
    test_file_path = './non-member_dataset.pkl'
    test_data = load_pkl_file(test_file_path)
    test_result = convert_data(test_data)
    
    plot2(train_result, test_result)
    
    
        