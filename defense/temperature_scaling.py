import torch
import torch.nn as nn
from datasets_utils.dataset_tools import collate_fn


class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_calibrated = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.temperature = 0

    def calibrate(self, validation_data, batch_size=32):
        #TODO: temperature calibration in object detection???
        self.model.eval()
        temperature = nn.Parameter(torch.tensor(1.0))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
        args = {'temperature': temperature}
        logits = []
        labels = []
        temperature_values = []
        loss_values = []

        data_loader =  torch.utils.data.DataLoader(validation_data, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, shuffle=False, pin_memory=True)

        with torch.no_grad():
            for images, boxes, labels, difficulties in data_loader:
                images = images.to(self.device)
                predicted_locs, predicted_scores = self.model(images) #(N, 8732, 4), (N, 8732, n_classes)
                logits.append(predicted_locs)                
                labels.append(labels)

        logits = torch.cat(logits, dim=0).to(self.device)
        labels = torch.cat(labels, dim=0).to(self.device)

        def T_scaling(logits, args):
            temperature = args.get('temperature', None)
            return torch.div(logits, temperature)

        def _eval():
            loss = criterion(T_scaling(logits, args), labels)
            loss.backward()
            temperature_values.append(temperature.item())
            loss_values.append(loss)
            return loss

        optimizer.step(_eval)
        self.temperature = temperature
        print('Best temperature value: {:.2f}'.format(self.temperature.item()))

    def calibrate_sgd(self, validation_data, batch_size=128):
        self.model.eval()
        temperature = nn.Parameter(torch.tensor(1.0))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([temperature], lr=0.01)
        max_iter = 100
        epsilon = 0.01
        data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)

        for iter in range(max_iter):
            old_temperature = temperature.item()
            for x, y in data_loader:
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = criterion(output / temperature, y)
                loss.backward()
                optimizer.step()
            if abs(old_temperature - temperature.item()) < epsilon:
                break
            print("iter: ",iter)
        self.temperature = temperature
        print('Best temperature value: {:.2f}'.format(self.temperature.item()))
        
    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            predicted_locs, predicted_scores = self.model(x)
            predicted_scores = predicted_scores / self.temperature
        return predicted_locs, predicted_scores
