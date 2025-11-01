import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from ImageOD.models import resnet   # adjust if your resnet file is elsewhere
from collections import OrderedDict

# Map names to constructors in your repo
net_name2object = {
    "resnet20_32x32": lambda: resnet.resnet20(num_classes=10),
    "resnet18_32x32": lambda: resnet.resnet18(num_classes=10),
    # add resnet34, wrn, etc. if needed
}

def test_id_accuracy(net, batch_size=200):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)) # CIFAR10 stats
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./openood/data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, choices=list(net_name2object.keys()),
                        help="Network architecture (e.g., resnet20_32x32, resnet18_32x32)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()

    # Build net
    net = net_name2object[args.net]()

    # Load checkpoint
    """
    state = torch.load(args.ckpt, map_location="cuda")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if "best_prec1" in state:
        state.pop("best_prec1", None)
    try:
        net.load_state_dict(state, strict=False)
    except Exception as e:
        print("Warning: some keys did not match when loading checkpoint:", e)
    """
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove "module." prefix if trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # strip "module." if present
        new_state_dict[new_key] = v

    net.load_state_dict(new_state_dict, strict=True)

    net = net.cuda()
    acc = test_id_accuracy(net)
    print(f"{args.net} CIFAR10 Test Accuracy: {acc:.2f}%")
