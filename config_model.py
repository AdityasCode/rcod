class config:
    def __init__(self):
        # basic model info for Landseer integration
        self.model_name = "resnet20_32x32"
        self.id_dataset = "cifar10"
        self.ood_dataset = "texture"
        self.checkpoint_path = "/app/robust-conformal-od/resnet20_cifar10.ckpt"
        self.postprocess = "react"
        self.n_train = 2000
        self.p_train = 0.03
