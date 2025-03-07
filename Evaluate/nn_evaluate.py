import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import warnings
import os


class Net256(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)

        self.conv_layers = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv2,
            self.bn2,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv3,
            self.bn3,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.dropout1,
        )

        input_size = (3, 256, 256)
        self.fc1_input_features = get_flattened_size(self, input_size)

        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_flattened_size(model, input_size):
    dummy_input = torch.zeros(1, *input_size)
    with torch.no_grad():
        output = model.conv_layers(dummy_input)
    return output.numel()


def predict_image_by_model(model_class, model_path, image_path, classes, img_size, mean_type, std_type):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_type, std=std_type)
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    return classes[pred]


def type_of_photo(file):
    classes1 = ['no_glasses', 'glasses']
    model_path1 = "../../models/NN_glasses_or_not.pt"
    file_path = file

    prediction = predict_image_by_model(Net256, model_path1, file_path, classes1, (256, 256), (0.5348, 0.4395, 0.3805),
                                        (0.2213, 0.1912, 0.1862))

    if prediction == classes1[0]:
        return 0

    else:
        classes2 = ['regular', 'sun']
        model_path2 = "../../models/NN_regular_or_sun.pt"

        prediction = predict_image_by_model(Net256, model_path2, file_path, classes2, (256, 256),
                                            (0.5037, 0.4416, 0.4057), (0.2471, 0.2342, 0.2316))

        if prediction == classes2[0]:
            return 1
        else:
            return 2


def main():
    warnings.filterwarnings("ignore")

    folder_paths = ["../../Evaluate_photos/no_glasses",
                    "../../Evaluate_photos/regular_glasses",
                    "../../Evaluate_photos/sunglasses"]
    print(folder_paths)

    correct = 0
    total = 0
    for i in range(len(folder_paths)):
        if not os.path.exists(folder_paths[i]):
            print(f"There is no folder: {folder_paths[i]}")

        for filename in os.listdir(folder_paths[i]):
            total += 1
            print(filename)

            image_path = os.path.join(folder_paths[i], filename)

            output = type_of_photo(image_path)
            print(output)

            if output == i:
                correct += 1

    print(f"\nTOTAL SCORE:\n{correct}/{total}")


if __name__ == '__main__':
    main()